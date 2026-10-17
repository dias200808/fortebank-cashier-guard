import time
from pathlib import Path

from app.schemas import JobResult


def stub_process(path: Path, job_id: str) -> JobResult:
    """First-deploy stand-in until the real pipeline (Task 15) is wired in."""
    time.sleep(2)
    return JobResult(job_id=job_id, status="done", verdict="no_violation", violations=[])
