import time

from app.jobs import JobStore, Worker
from app.schemas import JobResult


def wait_done(store, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        r = store.load(job_id)
        if r and r.status != "processing":
            return r
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_store_roundtrip(tmp_path):
    s = JobStore(tmp_path)
    job_id, path = s.new_job()
    assert path.name == "input.mov" and s.load(job_id).status == "processing"
    s.save(JobResult(job_id=job_id, status="done", verdict="no_violation"))
    assert s.load(job_id).verdict == "no_violation"


def test_load_rejects_path_traversal(tmp_path):
    assert JobStore(tmp_path).load("../etc") is None


def test_worker_processes_job(tmp_path):
    s = JobStore(tmp_path)
    w = Worker(s, lambda p, jid: JobResult(job_id=jid, status="done", verdict="no_violation"))
    w.start()
    job_id, path = s.new_job()
    path.write_bytes(b"x")
    w.submit(job_id)
    assert wait_done(s, job_id).status == "done"


def test_worker_survives_exception(tmp_path):
    s = JobStore(tmp_path)

    def process(p, jid):
        if p.read_bytes() == b"bad":
            raise ValueError("not a video")
        return JobResult(job_id=jid, status="done", verdict="no_violation")

    w = Worker(s, process)
    w.start()
    bad, bp = s.new_job(); bp.write_bytes(b"bad")
    good, gp = s.new_job(); gp.write_bytes(b"ok")
    w.submit(bad); w.submit(good)
    r_bad, r_good = wait_done(s, bad), wait_done(s, good)
    assert r_bad.status == "failed" and "not a video" in r_bad.error
    assert r_good.status == "done"


def test_unfinished_jobs_are_requeued_on_start(tmp_path):
    s = JobStore(tmp_path)
    job_id, path = s.new_job(); path.write_bytes(b"x")
    w = Worker(s, lambda p, jid: JobResult(job_id=jid, status="done", verdict="no_violation"))
    w.start()
    assert wait_done(s, job_id).status == "done"


def test_fail_marks_job(tmp_path):
    s = JobStore(tmp_path)
    job_id, _ = s.new_job()
    s.fail(job_id, "empty file")
    r = s.load(job_id)
    assert r.status == "failed" and r.error == "empty file"


def test_new_job_records_submission_time(tmp_path):
    s = JobStore(tmp_path)
    before = time.time()
    job_id, path = s.new_job()
    assert before <= float((path.parent / "submitted_at").read_text()) <= time.time()


def test_worker_survives_failed_save(tmp_path):
    class FlakyStore(JobStore):
        broken = True

        def save(self, result):
            if result.status == "done" and self.broken:
                self.broken = False
                raise OSError("disk full")
            super().save(result)

    s = FlakyStore(tmp_path)
    w = Worker(s, lambda p, jid: JobResult(job_id=jid, status="done", verdict="no_violation"))
    w.start()
    first, p1 = s.new_job(); p1.write_bytes(b"x")
    second, p2 = s.new_job(); p2.write_bytes(b"x")
    w.submit(first); w.submit(second)
    r1 = wait_done(s, first)
    assert r1.status == "failed" and "could not save result" in r1.error
    assert wait_done(s, second).status == "done"


def test_new_job_keeps_the_uploaded_file_name(tmp_path):
    s = JobStore(tmp_path)
    job_id, path = s.new_job("C:\\fakepath\\касса 1.mov")
    assert (path.parent / "filename").read_text(encoding="utf-8") == "касса 1.mov"
    bare, bare_path = s.new_job()
    assert not (bare_path.parent / "filename").exists()
    assert {j["job_id"]: j["filename"] for j in s.list_jobs()} == {job_id: "касса 1.mov", bare: None}


def test_clean_filename():
    from app.jobs import clean_filename
    assert clean_filename("a/b/../c.mov") == "c.mov"
    assert clean_filename("x\r\n\x00y.mov") == "xy.mov"
    assert clean_filename(None) == "" and clean_filename("   ") == ""
    assert len(clean_filename("v" * 500 + ".mov")) == 200


def test_history_does_not_parse_result_json(tmp_path, monkeypatch):
    from pathlib import Path
    from app.schemas import EvidenceFrame, Violation
    s = JobStore(tmp_path)
    job_id, _ = s.new_job("a.mov")
    big = [EvidenceFrame(timestamp_sec=1.0, image_base64="A" * 200_000)]
    s.save(JobResult(job_id=job_id, status="done", verdict="violation", violations=[
        Violation(type="t", timecode_sec=1.0, description="d", evidence_frames=big)] * 3))
    read = []
    orig = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: read.append(self.name) or orig(self, *a, **k))
    (j,) = s.list_jobs()
    assert j["verdict"] == "violation" and j["n_violations"] == 3 and j["status"] == "done"
    assert "result.json" not in read


def test_stale_summary_is_rebuilt(tmp_path):
    import json
    s = JobStore(tmp_path)
    job_id, path = s.new_job()
    # result.json replaced behind the store's back (e.g. a crash before the summary write)
    (path.parent / "result.json").write_text(JobResult(job_id=job_id, status="failed", error="boom").model_dump_json())
    (j,) = s.list_jobs()
    assert j["status"] == "failed" and j["error"] == "boom"
    (path.parent / "summary.json").write_text("{not json")
    assert s.summary(job_id)["status"] == "failed"
    assert json.loads((path.parent / "summary.json").read_text())["status"] == "failed"


def test_list_jobs_newest_first_and_limit(tmp_path):
    s = JobStore(tmp_path)
    ids = []
    for t in (100.0, 300.0, 200.0):
        job_id, path = s.new_job()
        (path.parent / "submitted_at").write_text(repr(t))
        ids.append(job_id)
    assert [j["job_id"] for j in s.list_jobs()] == [ids[1], ids[2], ids[0]]
    assert [j["job_id"] for j in s.list_jobs(limit=1)] == [ids[1]]
    assert s.list_jobs()[0]["submitted_at"] == "1970-01-01T00:05:00+00:00"


def test_duration_is_listed_when_known(tmp_path):
    s = JobStore(tmp_path)
    job_id, _ = s.new_job()
    assert s.list_jobs()[0]["duration_s"] is None
    s.set_duration(job_id, 84.236)
    assert s.list_jobs()[0]["duration_s"] == 84.2


def test_unfinished_in_submission_order_and_orphans_failed(tmp_path):
    s = JobStore(tmp_path)
    late, lp = s.new_job(); lp.write_bytes(b"x"); (lp.parent / "submitted_at").write_text("200.0")
    early, ep = s.new_job(); ep.write_bytes(b"x"); (ep.parent / "submitted_at").write_text("100.0")
    orphan, _ = s.new_job()  # no video was ever saved
    assert s.unfinished() == [early, late]
    r = s.load(orphan)
    assert r.status == "failed" and "interrupted" in r.error


def test_worker_reports_waiting_jobs(tmp_path):
    import threading
    gate, started = threading.Event(), threading.Event()

    def slow(p, jid):
        started.set()
        gate.wait(5)
        return JobResult(job_id=jid, status="done", verdict="no_violation")

    s = JobStore(tmp_path)
    w = Worker(s, slow)
    w.start()
    a, ap = s.new_job(); ap.write_bytes(b"x")
    b, bp = s.new_job(); bp.write_bytes(b"x")
    w.submit(a)
    assert started.wait(5)
    w.submit(b)
    assert w.waiting() == {b}
    gate.set()
    wait_done(s, a); wait_done(s, b)
    assert w.waiting() == set()
