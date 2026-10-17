import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from app.schemas import JobResult

ProcessFn = Callable[[Path, str], JobResult]
log = logging.getLogger(__name__)


class JobStore:
    """One directory per job: input.mov + result.json (written atomically) + submitted_at (unix time)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_job(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        (self.root / job_id).mkdir()
        (self.root / job_id / "submitted_at").write_text(repr(time.time()), encoding="utf-8")
        self.save(JobResult(job_id=job_id, status="processing"))
        return job_id, self.video_path(job_id)

    def video_path(self, job_id: str) -> Path:
        return self.root / job_id / "input.mov"

    def save(self, result: JobResult) -> None:
        d = self.root / result.job_id
        tmp = d / "result.json.tmp"
        tmp.write_text(result.model_dump_json(), encoding="utf-8")
        tmp.replace(d / "result.json")

    def fail(self, job_id: str, message: str) -> None:
        self.save(JobResult(job_id=job_id, status="failed", error=message))

    def load(self, job_id: str) -> JobResult | None:
        if not job_id.isalnum():
            return None
        p = self.root / job_id / "result.json"
        if not p.exists():
            return None
        return JobResult.model_validate_json(p.read_text(encoding="utf-8"))

    def unfinished(self) -> list[str]:
        ids = []
        for p in sorted(self.root.glob("*/result.json")):
            r = JobResult.model_validate_json(p.read_text(encoding="utf-8"))
            if r.status == "processing" and self.video_path(r.job_id).exists():
                ids.append(r.job_id)
        return ids


class Worker:
    """Single GPU worker: jobs run one at a time in submission order."""

    def __init__(self, store: JobStore, process: ProcessFn):
        self.store, self.process = store, process
        self.q: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        for job_id in self.store.unfinished():
            self.q.put(job_id)
        self._thread.start()

    def submit(self, job_id: str) -> None:
        self.q.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self.q.get()
            try:
                self._run(job_id)
            except Exception:  # never let the worker thread die
                log.exception("job %s: unexpected worker error", job_id)
            finally:
                self.q.task_done()

    def _run(self, job_id: str) -> None:
        try:
            result = self.process(self.store.video_path(job_id), job_id)
        except Exception as e:  # one bad video must not stop the worker
            log.exception("job %s failed", job_id)
            result = JobResult(job_id=job_id, status="failed", error=f"{type(e).__name__}: {e}")
        try:
            self.store.save(result)
        except Exception as e:
            log.exception("job %s: could not save result", job_id)
            try:
                self.store.fail(job_id, f"could not save result: {type(e).__name__}: {e}")
            except Exception:
                log.exception("job %s: could not mark the job failed", job_id)
