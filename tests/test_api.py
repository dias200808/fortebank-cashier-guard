import time
from dataclasses import replace

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.schemas import JobResult

KEY = {"X-API-Key": "k"}


def make_client(tmp_path, process=None):
    s = replace(Settings(), api_key="k", jobs_dir=tmp_path)
    process = process or (lambda p, jid: JobResult(job_id=jid, status="done", verdict="no_violation"))
    return TestClient(create_app(s, process))


def poll(c, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        d = c.get(f"/api/v1/jobs/{job_id}", headers=KEY).json()
        if d["status"] != "processing":
            return d
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_requires_key(tmp_path):
    with make_client(tmp_path) as c:
        assert c.post("/api/v1/analyze", files={"file": ("a.mov", b"x")}).status_code == 401
        assert c.post("/api/v1/analyze", files={"file": ("a.mov", b"x")}, headers={"X-API-Key": "bad"}).status_code == 401
        assert c.get("/api/v1/jobs/abc").status_code == 401


def test_analyze_then_poll(tmp_path):
    with make_client(tmp_path) as c:
        r = c.post("/api/v1/analyze", files={"file": ("a.mov", b"video")}, headers=KEY)
        assert r.status_code == 200
        d = poll(c, r.json()["job_id"])
        assert d["status"] == "done" and d["verdict"] == "no_violation" and d["violations"] == []


def test_unknown_job_404(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/api/v1/jobs/deadbeef0000", headers=KEY).status_code == 404


def test_empty_upload_rejected(tmp_path):
    with make_client(tmp_path) as c:
        r = c.post("/api/v1/analyze", files={"file": ("a.mov", b"")}, headers=KEY)
        assert r.status_code == 400


def test_five_parallel_uploads_all_finish(tmp_path):
    with make_client(tmp_path) as c:
        ids = [c.post("/api/v1/analyze", files={"file": (f"{i}.mov", b"v")}, headers=KEY).json()["job_id"] for i in range(5)]
        assert all(poll(c, i)["status"] == "done" for i in ids)


def test_index_served(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/").status_code == 200


def test_index_has_verdict_strings_and_key_header(tmp_path):
    with make_client(tmp_path) as c:
        body = c.get("/").text
        assert "Нарушение не обнаружено" in body
        assert "Обнаружены признаки возможных нарушений" in body
        assert "X-API-Key" in body


VIDEO = bytes(range(256)) * 4  # 1024 bytes of distinct-ish content


def upload(c, data=VIDEO):
    r = c.post("/api/v1/analyze", files={"file": ("a.mov", data)}, headers=KEY)
    assert r.status_code == 200
    return r.json()["job_id"]


def test_video_requires_key(tmp_path):
    with make_client(tmp_path) as c:
        jid = upload(c)
        assert c.get(f"/api/v1/jobs/{jid}/video").status_code == 401
        assert c.get(f"/api/v1/jobs/{jid}/video", params={"key": "bad"}).status_code == 401
        assert c.get(f"/api/v1/jobs/{jid}/video", headers={"X-API-Key": "bad"}).status_code == 401


def test_video_with_header_key(tmp_path):
    with make_client(tmp_path) as c:
        jid = upload(c)
        r = c.get(f"/api/v1/jobs/{jid}/video", headers=KEY)
        assert r.status_code == 200
        assert r.content == VIDEO
        assert r.headers["content-type"].startswith("video/quicktime")
        assert r.headers["accept-ranges"] == "bytes"


def test_video_with_query_key(tmp_path):
    with make_client(tmp_path) as c:
        jid = upload(c)
        r = c.get(f"/api/v1/jobs/{jid}/video", params={"key": "k"})
        assert r.status_code == 200 and r.content == VIDEO


def test_video_range_returns_partial_content(tmp_path):
    with make_client(tmp_path) as c:
        jid = upload(c)
        r = c.get(f"/api/v1/jobs/{jid}/video", headers={**KEY, "Range": "bytes=0-99"})
        assert r.status_code == 206
        assert r.headers["content-range"] == f"bytes 0-99/{len(VIDEO)}"
        assert r.headers["content-length"] == "100"
        assert r.content == VIDEO[:100]


def test_video_range_with_query_key(tmp_path):
    with make_client(tmp_path) as c:
        jid = upload(c)
        r = c.get(f"/api/v1/jobs/{jid}/video", params={"key": "k"}, headers={"Range": "bytes=1000-"})
        assert r.status_code == 206
        assert r.headers["content-range"] == f"bytes 1000-1023/{len(VIDEO)}"
        assert r.content == VIDEO[1000:]


def test_video_unknown_job(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/api/v1/jobs/deadbeef0000/video", headers=KEY).status_code == 404
        assert c.get("/api/v1/jobs/deadbeef0000/video", params={"key": "k"}).status_code == 404
        assert c.get("/api/v1/jobs/deadbeef0000/video").status_code == 401
        assert c.get("/api/v1/jobs/..%2Fetc/video", headers=KEY).status_code == 404


def test_job_endpoint_still_header_only(tmp_path):
    with make_client(tmp_path) as c:
        jid = upload(c)
        assert c.get(f"/api/v1/jobs/{jid}", params={"key": "k"}).status_code == 401
