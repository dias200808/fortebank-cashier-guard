import json
import logging
import math
import os
import queue
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.schemas import JobResult

ProcessFn = Callable[[Path, str], JobResult]
log = logging.getLogger(__name__)

SUMMARY = "summary.json"  # status/verdict/count of result.json, so the history never parses results with images
SUMMARY_FIELDS = {"status", "verdict", "n_violations", "error"}
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def clean_filename(name: str | None) -> str:
    """Base name of the client's file name (old browsers send a full Windows path), no control characters."""
    return CONTROL_CHARS.sub("", (name or "").replace("\\", "/").rsplit("/", 1)[-1]).strip()[:200]


def summarize(r: dict) -> dict:
    """The history fields of a result (a JobResult's fields or its parsed JSON)."""
    status, verdict = r.get("status"), r.get("verdict")
    n = len(r.get("violations") or []) if status == "done" else None
    if status == "done" and verdict is None:
        verdict = "violation" if n else "no_violation"
    return {"status": status, "verdict": verdict if status == "done" else None, "n_violations": n,
            "error": r.get("error") if status == "failed" else None}


def _stat_key(p: Path) -> list[int]:
    st = p.stat()
    return [st.st_mtime_ns, st.st_size]


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except (OSError, ValueError):
        return None


def _read_float(p: Path) -> float | None:
    try:
        x = float(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


class JobStore:
    """One directory per job: input.mov + result.json (written atomically) + submitted_at (unix time)
    + filename (name of the uploaded file) + duration_s (once probed) + summary.json (history cache of result.json)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_job(self, filename: str | None = None) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        d = self.root / job_id
        d.mkdir()
        (d / "submitted_at").write_text(repr(time.time()), encoding="utf-8")
        if name := clean_filename(filename):
            (d / "filename").write_text(name, encoding="utf-8")
        self.save(JobResult(job_id=job_id, status="processing"))
        return job_id, self.video_path(job_id)

    def discard(self, job_id: str) -> None:
        """Delete a job whose upload was rejected: the client got no job_id, so nothing can ask for it."""
        if job_id.isalnum():
            shutil.rmtree(self.root / job_id, ignore_errors=True)

    def video_path(self, job_id: str) -> Path:
        return self.root / job_id / "input.mov"

    def save(self, result: JobResult) -> None:
        d = self.root / result.job_id
        tmp = d / "result.json.tmp"
        tmp.write_text(result.model_dump_json(), encoding="utf-8")
        tmp.replace(d / "result.json")
        self._write_summary(d, summarize({"status": result.status, "verdict": result.verdict,
                                          "violations": result.violations, "error": result.error}))

    def _write_summary(self, d: Path, summary: dict, key: list[int] | None = None) -> None:
        """Best effort: the summary is a cache keyed by result.json's mtime and size, a failed write only costs a
        re-parse later. A unique temp name, as the worker and a history request may write it at the same time."""
        tmp = d / f"{SUMMARY}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            key = key or _stat_key(d / "result.json")
            tmp.write_text(json.dumps({**summary, "key": key}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(d / SUMMARY)
        except OSError as e:
            log.warning("%s: could not write %s (%s)", d.name, SUMMARY, e)
            tmp.unlink(missing_ok=True)

    def summary(self, job_id: str) -> dict | None:
        """summarize() of the job's result.json without parsing it while summary.json matches its mtime and size.
        Jobs saved before summaries existed (or a crash between the two writes) are parsed once and cached."""
        if not job_id.isalnum():
            return None
        d = self.root / job_id
        try:
            key = _stat_key(d / "result.json")
        except OSError:
            return None
        try:
            s = json.loads((d / SUMMARY).read_text(encoding="utf-8"))
            if s.pop("key") == key and set(s) == SUMMARY_FIELDS:
                return s
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            pass
        try:
            raw = json.loads((d / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        s = summarize(raw)
        self._write_summary(d, s, key)
        return s

    def submitted(self, job_id: str) -> float | None:
        d = self.root / job_id
        ts = _read_float(d / "submitted_at")
        if ts is None:
            try:
                ts = (d / "result.json").stat().st_mtime
            except OSError:
                return None
        return ts

    def set_duration(self, job_id: str, seconds: float) -> None:
        (self.root / job_id / "duration_s").write_text(f"{seconds:.3f}", encoding="utf-8")

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """History, newest first. Per job only a few tiny files are read, never the evidence images."""
        found = []
        try:
            with os.scandir(self.root) as it:
                for e in it:
                    if e.name.isalnum() and e.is_dir() and (ts := self.submitted(e.name)) is not None:
                        found.append((ts, e.name))
        except OSError:
            return []
        jobs = []
        for ts, job_id in sorted(found, reverse=True):
            if len(jobs) >= limit:
                break
            s = self.summary(job_id)
            if s is None:
                continue
            d = self.root / job_id
            dur = _read_float(d / "duration_s")
            jobs.append({"job_id": job_id, "filename": _read_text(d / "filename"), "status": s["status"],
                         "verdict": s["verdict"], "n_violations": s["n_violations"],
                         "submitted_at": datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds"),
                         "duration_s": round(dur, 1) if dur else None, "error": s["error"]})
        return jobs

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
        """Jobs a restart left in "processing", in submission order. One whose video was never saved cannot run:
        it is marked failed instead of staying "processing" in the history forever."""
        found = []
        for p in self.root.glob("*/result.json"):
            job_id = p.parent.name
            s = self.summary(job_id)
            if not s or s["status"] != "processing":
                continue
            if self.video_path(job_id).exists():
                found.append((self.submitted(job_id) or 0.0, job_id))
            else:
                self.fail(job_id, "interrupted before the video was saved")
        return [job_id for _, job_id in sorted(found)]


class Worker:
    """Single GPU worker: jobs run one at a time in submission order."""

    def __init__(self, store: JobStore, process: ProcessFn):
        self.store, self.process = store, process
        self.q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._waiting: list[str] = []  # submitted, not started yet: the history shows them as queued
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        for job_id in self.store.unfinished():
            self.submit(job_id)
        self._thread.start()

    def submit(self, job_id: str) -> None:
        with self._lock:
            self._waiting.append(job_id)
        self.q.put(job_id)

    def waiting(self) -> set[str]:
        with self._lock:
            return set(self._waiting)

    def _loop(self) -> None:
        while True:
            job_id = self.q.get()
            with self._lock:
                if job_id in self._waiting:
                    self._waiting.remove(job_id)
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
