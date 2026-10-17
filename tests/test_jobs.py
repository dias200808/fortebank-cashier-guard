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
