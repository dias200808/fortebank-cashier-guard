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


# ---------- jury contract (TZ section 10), field by field ----------

JPEG_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD"


def violating(p, jid):
    from app.schemas import EvidenceFrame, RuleRef, Violation
    v = Violation(type="Наличные оставлены без присмотра", timecode_sec=44.2, description="Кассир отошёл от стола",
                  evidence_frames=[EvidenceFrame(timestamp_sec=44.2, image_base64=JPEG_B64)],
                  rule_id="R01", rule=RuleRef(act="Правила", point="п. 20", strength="В"))
    return JobResult(job_id=jid, status="done", verdict="violation", violations=[v])


def test_contract_analyze_response(tmp_path):
    with make_client(tmp_path) as c:
        r = c.post("/api/v1/analyze", files={"file": ("video_017.mov", VIDEO, "video/quicktime")}, headers=KEY)
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
        d = r.json()
        assert set(d) == {"job_id", "status"} and d["status"] == "processing"
        assert isinstance(d["job_id"], str) and d["job_id"]


def test_contract_multipart_field_is_named_file(tmp_path):
    with make_client(tmp_path) as c:
        assert c.post("/api/v1/analyze", files={"video": ("a.mov", VIDEO)}, headers=KEY).status_code == 422


def test_contract_key_header_is_case_insensitive(tmp_path):
    with make_client(tmp_path) as c:
        r = c.post("/api/v1/analyze", files={"file": ("a.mov", VIDEO)}, headers={"x-api-key": "k"})
        assert r.status_code == 200
        assert c.get(f"/api/v1/jobs/{r.json()['job_id']}", headers={"X-Api-Key": "k"}).status_code == 200


def test_contract_401_without_valid_key_everywhere(tmp_path):
    with make_client(tmp_path) as c:
        jid = upload(c)
        for headers in ({}, {"X-API-Key": ""}, {"X-API-Key": "K"}, {"X-API-Key": "k "}):
            assert c.post("/api/v1/analyze", files={"file": ("a.mov", VIDEO)}, headers=headers).status_code == 401
            assert c.get(f"/api/v1/jobs/{jid}", headers=headers).status_code == 401
            assert c.get("/api/v1/jobs", headers=headers).status_code == 401
            assert c.get(f"/api/v1/jobs/{jid}/video", headers=headers).status_code == 401
        assert c.post("/api/v1/analyze", headers={}).status_code in (401, 422)  # no file and no key: never accepted


def test_contract_processing_shape(tmp_path):
    import threading
    gate = threading.Event()

    def slow(p, jid):
        gate.wait(5)
        return JobResult(job_id=jid, status="done", verdict="no_violation")

    with make_client(tmp_path, slow) as c:
        jid = upload(c)
        assert c.get(f"/api/v1/jobs/{jid}", headers=KEY).json() == {"job_id": jid, "status": "processing"}
        gate.set()
        assert poll(c, jid)["status"] == "done"


def test_contract_done_violation_shape(tmp_path):
    with make_client(tmp_path, violating) as c:
        jid = upload(c)
        d = poll(c, jid)
        assert d["job_id"] == jid and d["status"] == "done" and d["verdict"] == "violation"
        assert isinstance(d["violations"], list) and len(d["violations"]) == 1
        v = d["violations"][0]
        assert isinstance(v["type"], str) and v["type"]
        assert isinstance(v["timecode_sec"], float) and v["timecode_sec"] == 44.2
        assert isinstance(v["description"], str) and v["description"]
        assert 1 <= len(v["evidence_frames"]) <= 3
        for f in v["evidence_frames"]:
            assert set(f) == {"timestamp_sec", "image_base64"}
            assert isinstance(f["timestamp_sec"], float) and f["image_base64"].startswith("/9j/")
        assert None not in v.values()  # optional extras are omitted, never null
        assert "hints" not in d


def test_contract_no_violation_has_empty_list(tmp_path):
    with make_client(tmp_path) as c:
        d = poll(c, upload(c))
        assert d["verdict"] == "no_violation" and d["violations"] == []


def test_contract_failed_shape(tmp_path):
    def broken(p, jid):
        raise ValueError("not a video")

    with make_client(tmp_path, broken) as c:
        d = poll(c, upload(c))
        assert d["status"] == "failed" and isinstance(d["error"], str) and "not a video" in d["error"]
        assert "violations" not in d and "verdict" not in d


# ---------- GET /api/v1/jobs: history for the UI ----------

def wait_status(c, timeout=5.0, **want):
    end = time.time() + timeout
    while time.time() < end:
        jobs = c.get("/api/v1/jobs", headers=KEY).json()["jobs"]
        if all(j["status"] != "processing" for j in jobs) and all(
                all(j.get(k) == v for k, v in want.items()) for j in jobs):
            return jobs
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_list_jobs_requires_key_header_or_query(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/api/v1/jobs").status_code == 401
        assert c.get("/api/v1/jobs", params={"key": "bad"}).status_code == 401
        assert c.get("/api/v1/jobs", headers=KEY).json() == {"jobs": []}
        assert c.get("/api/v1/jobs", params={"key": "k"}).json() == {"jobs": []}


def test_list_jobs_newest_first_with_fields(tmp_path):
    from datetime import datetime

    def process(p, jid):
        return violating(p, jid) if p.read_bytes() == b"bad" else JobResult(job_id=jid, status="done", verdict="no_violation")

    with make_client(tmp_path, process) as c:
        ids = []
        for name, data in (("first.mov", b"ok"), ("C:\\Users\\cam\\второе видео.mov", b"bad"), ("third.mov", b"ok")):
            ids.append(c.post("/api/v1/analyze", files={"file": (name, data)}, headers=KEY).json()["job_id"])
            time.sleep(0.01)
        jobs = wait_status(c)
        assert [j["job_id"] for j in jobs] == ids[::-1]
        third, second, first = jobs
        assert set(first) == {"job_id", "filename", "status", "verdict", "n_violations", "submitted_at", "duration_s",
                              "error", "queued"}
        assert first["filename"] == "first.mov" and second["filename"] == "второе видео.mov"
        assert second["status"] == "done" and second["verdict"] == "violation" and second["n_violations"] == 1
        assert first["verdict"] == "no_violation" and first["n_violations"] == 0
        assert first["error"] is None and first["queued"] is False and first["duration_s"] is None  # not a real video
        assert datetime.fromisoformat(first["submitted_at"]).tzinfo is not None
        assert first["submitted_at"] <= second["submitted_at"] <= third["submitted_at"]


def test_list_jobs_limit(tmp_path):
    with make_client(tmp_path) as c:
        ids = [upload(c) for _ in range(4)]
        wait_status(c)
        assert [j["job_id"] for j in c.get("/api/v1/jobs", params={"limit": 2}, headers=KEY).json()["jobs"]] == ids[:1:-1]
        assert len(c.get("/api/v1/jobs", params={"limit": 0}, headers=KEY).json()["jobs"]) == 1     # clamped to 1..200
        assert len(c.get("/api/v1/jobs", params={"limit": 9999}, headers=KEY).json()["jobs"]) == 4
        assert c.get("/api/v1/jobs", params={"limit": "x"}, headers=KEY).status_code == 422


def test_list_jobs_failed_job_has_error(tmp_path):
    def broken(p, jid):
        raise RuntimeError("decoder exploded")

    with make_client(tmp_path, broken) as c:
        upload(c)
        (j,) = wait_status(c)
        assert j["status"] == "failed" and "decoder exploded" in j["error"]
        assert j["verdict"] is None and j["n_violations"] is None


def test_list_jobs_marks_queued_and_running(tmp_path):
    import threading
    gate, started = threading.Event(), threading.Event()

    def slow(p, jid):
        started.set()
        gate.wait(5)
        return JobResult(job_id=jid, status="done", verdict="no_violation")

    with make_client(tmp_path, slow) as c:
        running = upload(c)
        assert started.wait(5)
        queued = upload(c)
        jobs = {j["job_id"]: j for j in c.get("/api/v1/jobs", headers=KEY).json()["jobs"]}
        assert jobs[running]["status"] == "processing" and jobs[running]["queued"] is False
        assert jobs[queued]["status"] == "processing" and jobs[queued]["queued"] is True
        gate.set()
        assert all(j["queued"] is False for j in wait_status(c))


def test_rejected_uploads_leave_no_job(tmp_path):
    s = replace(Settings(), api_key="k", jobs_dir=tmp_path, max_upload_mb=0)
    with TestClient(create_app(s, lambda p, jid: JobResult(job_id=jid, status="done", verdict="no_violation"))) as c:
        assert c.post("/api/v1/analyze", files={"file": ("a.mov", b"")}, headers=KEY).status_code == 400
        assert c.post("/api/v1/analyze", files={"file": ("a.mov", VIDEO)}, headers=KEY).status_code == 413
        assert c.get("/api/v1/jobs", headers=KEY).json() == {"jobs": []}
        assert list(tmp_path.iterdir()) == []


def test_list_jobs_reports_video_duration(tmp_path):
    from tests.conftest import write_video
    import numpy as np
    data = write_video(tmp_path / "src.mov", 20, 10, lambda i: np.full((96, 160, 3), i * 10, np.uint8)).read_bytes()
    with make_client(tmp_path / "jobs") as c:
        jid = c.post("/api/v1/analyze", files={"file": ("cam.mov", data)}, headers=KEY).json()["job_id"]
        end = time.time() + 10
        while time.time() < end:
            (j,) = c.get("/api/v1/jobs", headers=KEY).json()["jobs"]
            if j["duration_s"] is not None:
                break
            time.sleep(0.05)
        assert j["job_id"] == jid and abs(j["duration_s"] - 2.0) <= 0.2


def test_list_jobs_old_layout_without_filename_or_summary(tmp_path):
    import json
    old = tmp_path / "0123456789ab"
    old.mkdir()
    (old / "submitted_at").write_text("1760000000.0", encoding="utf-8")
    (old / "input.mov").write_bytes(b"x")
    (old / "result.json").write_text(violating(None, "0123456789ab").model_dump_json(), encoding="utf-8")
    (tmp_path / "not-a-job").mkdir()
    with make_client(tmp_path) as c:
        (j,) = c.get("/api/v1/jobs", headers=KEY).json()["jobs"]
        assert j["job_id"] == "0123456789ab" and j["filename"] is None and j["verdict"] == "violation"
        assert j["n_violations"] == 1 and j["submitted_at"].startswith("2025-10-09T")
    assert json.loads((old / "summary.json").read_text())["n_violations"] == 1  # cached for the next request
