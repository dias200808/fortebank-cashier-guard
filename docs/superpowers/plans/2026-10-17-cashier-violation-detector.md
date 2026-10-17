# Cashier Violation Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local service (web UI + API with `X-API-Key`) that takes a cashier-room `.mov` and returns verdict, violations (type, timecode, 1–3 evidence JPEGs, description, Kazakhstan regulation clause) within 2 minutes.

**Architecture:** CV journal (person detector + ByteTrack → cashier/visitor events with precise times) + VLM yes/no fact probes over the whole video at 1 fps on zone crops (P(yes) series) → smoothed state intervals → rule catalog (`rules.yaml`) produces candidates → VLM verification on short clips → report builder takes type/clause from the catalog and timecode from the CV journal.

**Tech Stack:** Python 3.12, FastAPI, PyAV, OpenCV, torchvision Faster R-CNN (BSD-3), `trackers` ByteTrack (Apache-2.0), `supervision` (MIT), vLLM serving Qwen3-VL (Apache-2.0) via OpenAI-compatible API, pytest.

**Spec:** `docs/methodology.md` (approved 2026-10-17). Norm texts: `docs/normative_research.md`. Dataset notes: `docs/dataset_analysis.md`.

## Global Constraints

- Only local models; no cloud AI APIs during processing (TZ §7, FAQ 35). After weights are downloaded, `HF_HUB_OFFLINE=1`.
- Licenses must allow commercial use: no Ultralytics/YOLO-World/VideoMAE/VideoLLaMA3 (FAQ 24).
- No face recognition, no audio (FAQ 28–29).
- API: `POST /api/v1/analyze` (multipart field `file`) → `{"job_id": ...}`; `GET /api/v1/jobs/{job_id}` → `status` ∈ `processing|done|failed`; result keys `job_id, status, verdict (violation|no_violation), violations[{type, timecode_sec, description, evidence_frames[{timestamp_sec, image_base64}]}]`. Missing/wrong `X-API-Key` → 401.
- 1–3 evidence frames per violation, JPEG base64.
- Regulation references come only from `app/rules.yaml`; the VLM never writes clause numbers.
- VLM never supplies timecodes; times come from the CV journal or state-interval starts.
- Processing target ≤ 2 min per video on RTX 3090; jury limit 5 min.
- No git yet: every "Checkpoint" step = run `pytest -q` and fix failures. Once the repo exists, commit after each task.
- All paths below are relative to `/Users/dastan.akhmetbek/Desktop/datathon` (the project root).

## Review Focus

1. Phone video with rotation metadata or different resolution/fps (e.g. 1080×1920, 60 fps) → frames upright, zones relative, timecodes in seconds from stream start. Pinned in Task 5 (`test_apply_rotation_clockwise`, `test_timestamps_start_at_zero_with_offset_stream`).
2. Upload that is empty, not a video, or truncated → job ends `failed` with an error message; the worker keeps serving the next job. Pinned in Task 1 (`test_worker_survives_exception`) and Task 2 (`test_empty_upload_rejected`), Task 15 (`test_process_rejects_non_video`).
3. VLM server down or timing out → bounded retries, job `failed` with a clear error well before 5 min, no hang. Pinned in Task 9 (`test_retries_then_raises`).
4. Long video (145 s) or slow VLM → time budget keeps the job under the limit (skip verification, mark probe-strong candidates). Pinned in Task 15 (`test_budget_skips_verification`).
5. Video with no people at all / camera covered → no crash; no cashier events; camera-covered rule may fire. Pinned in Task 7 (`test_no_tracks`) and Task 13 (`test_quality_rule_dark_video`).

## Evidence that shaped the plan (probe, 2026-10-17, 40 videos, CPU)

- Raw tracker events (appear/disappear/count) hit 20/21 labels within ±3 s but cover ~90 % of video time → useless as a filter. Hence: VLM probes cover the whole video at 1 fps; CV events are for timing only.
- Reflections in the glass partition are tracked as people (box height 0.06–0.09 of frame vs 0.13–0.23 for people in the room) → keep tracks with median height ≥ 0.5 × cashier height.
- Hand-only actions (X-1, X-2, X-4@68, X-18) produce no tracker event and no masked scene change → added `F_handover` fact and an unmasked desk "hand activity" signal that adds dense probe frames.
- "Cashier left" fires late if defined as "far from seat": labels mark the start of the movement. Departure = away chain (gaps ≤ 3 s) that ends with the cashier gone, timestamped at chain start; trips to the safe zone are not departures.

## Deferred from the spec (decided while planning)

- R07 (смешение наличных), R16 (упаковка без контролёра), R19 (выдача без пересчёта): need hand-level detail that 464×848 frames do not reliably show; listed in README limitations instead of building rules with no test-backed signal.
- Manual zone drawing in the UI (spec §13 fallback): not built; if S1 misses a zone, the facts for that zone fall back to the full frame.
- Spec S1 named Florence-2 first; the plan uses the served Qwen3-VL for grounding to keep one model server (Florence-2 stays a fallback idea).
- Spec §9 "Docker image": Vast.ai instances are already containers, so `deploy/setup.sh` + `deploy/start.sh` replace a Dockerfile.

## File Structure

```
requirements.txt               # pinned deps for app (vLLM lives in its own venv on the GPU host)
pytest.ini
app/
  __init__.py
  config.py                    # Settings dataclass + load_settings() from env
  schemas.py                   # pydantic API models (JobResult, Violation, EvidenceFrame, RuleRef)
  jobs.py                      # JobStore (disk) + Worker (queue thread)
  api.py                       # create_app(settings, process) — FastAPI routes
  main.py                      # uvicorn entry: builds settings, processor, app
  stub.py                      # stub processor for the first deploy (fixed no_violation)
  facts.yaml                   # VLM fact questions (S4)
  rules.yaml                   # rule catalog: type, family, facts, params, norm, template, verify
  static/index.html            # one-page UI
  pipeline/
    __init__.py
    geom.py                    # normalized Box helpers, crop
    frames.py                  # S0 decode → VideoFrames
    people.py                  # S2 detector, tracking, cashier/visitor events
    change.py                  # S3 per-zone change & hand activity
    vlm.py                     # VLM protocol, OpenAIVLM (vLLM), FakeVLM
    scene.py                   # S1 zones via VLM grounding
    probes.py                  # S4 fact probes
    timeline.py                # S5 smoothing + hysteresis intervals
    rules.py                   # S6 rule loading + evaluation
    verify.py                  # S7 clip verification
    report.py                  # S9 merge, evidence frames, descriptions
    run.py                     # orchestrator analyze() + build_processor()
scripts/
  __init__.py
  labels.py                    # parse open/open_labels.xlsx (dedupe X-47→X-4, X-48→X-5)
  eval.py                      # run pipeline on dataset, metrics per layout
  check_grounding.py           # find the VLM bbox coordinate convention
data/
  types.csv                    # our hypothesised rule id per labelled violation
  layouts.csv                  # camera layout per video (A–E)
deploy/
  setup.sh                     # GPU host setup (venvs, weights)
  start.sh                     # start vLLM + API
tests/
  conftest.py                  # synthetic video writer + helpers
  test_*.py                    # one file per module
README.md                      # submission doc
```

## Team tracks (run in parallel after Task 0)

- **Track A — service & deploy (person 1):** Tasks 1 → 2 → 3 → 4. Stub service on a public URL by end of track.
- **Track B — perception (person 2):** Tasks 5 → 6 → 7 → 8.
- **Track C — VLM & logic (person 3):** Tasks 9 → 10 → 11 → 12 → 13 → 14.
- **Integration (all):** Task 15 (needs 5–14) → Task 16 (eval + calibration on GPU host) → Tasks 17, 18 (stretch) → Task 19 (docs, morning 18.10).

---

### Task 0: Project skeleton and test harness

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `app/__init__.py`, `app/pipeline/__init__.py`, `scripts/__init__.py`, `tests/conftest.py`, `tests/test_conftest.py`

**Interfaces:**
- Produces: `tests.conftest.write_video(path, n_frames, fps, painter, size=(96,160)) -> Path`; fixture `tmp_video(tmp_path)` returning a factory with the same signature minus `path`.

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
pydantic>=2.7
av>=12
numpy>=1.26
opencv-python-headless>=4.9
pillow>=10
torch>=2.4
torchvision>=0.19
supervision>=0.25
trackers>=2.6
httpx>=0.27
pyyaml>=6
pytest>=8
```

- [ ] **Step 2: Create `pytest.ini` and empty packages**

```ini
[pytest]
testpaths = tests
addopts = -q
```

Run: `mkdir -p app/pipeline app/static scripts tests data deploy && touch app/__init__.py app/pipeline/__init__.py scripts/__init__.py`

- [ ] **Step 3: Write `tests/conftest.py`**

```python
from pathlib import Path
from typing import Callable

import av
import numpy as np
import pytest

Painter = Callable[[int], np.ndarray]


def write_video(path: Path, n_frames: int, fps: int, painter: Painter, size: tuple[int, int] = (96, 160)) -> Path:
    """Write an mpeg4 .mov whose frame i is painter(i) (H x W x 3 uint8, W x H = size)."""
    w, h = size
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for i in range(n_frames):
        frame = av.VideoFrame.from_ndarray(painter(i), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture
def tmp_video(tmp_path):
    def make(n_frames: int, fps: int, painter: Painter, size=(96, 160), name="v.mov") -> Path:
        return write_video(tmp_path / name, n_frames, fps, painter, size)
    return make
```

- [ ] **Step 4: Write the smoke test `tests/test_conftest.py`**

```python
import av
import numpy as np


def test_write_video_roundtrip(tmp_video):
    path = tmp_video(30, 30, lambda i: np.full((160, 96, 3), 128, np.uint8))
    c = av.open(str(path))
    frames = list(c.decode(c.streams.video[0]))
    assert len(frames) == 30
    assert abs(frames[0].to_ndarray(format="rgb24").mean() - 128) < 10
```

- [ ] **Step 5: Install and run**

Run: `.venv/bin/pip install -r requirements.txt && .venv/bin/pytest -q`
Expected: `1 passed`

- [ ] **Step 6: Checkpoint** — `pytest -q` green.

---

### Task 1: Schemas, job store and worker (Track A)

**Files:**
- Create: `app/config.py`, `app/schemas.py`, `app/jobs.py`
- Test: `tests/test_schemas.py`, `tests/test_jobs.py`

**Interfaces:**
- Produces: `Settings`, `load_settings()`; `EvidenceFrame`, `RuleRef`, `Violation`, `JobResult.to_api() -> dict`; `JobStore(root).new_job() -> (job_id, Path)`, `.save(JobResult)`, `.load(job_id) -> JobResult|None`, `.unfinished() -> list[str]`, `.fail(job_id, msg)`; `Worker(store, process).start()`, `.submit(job_id)`, `.q`; `ProcessFn = Callable[[Path, str], JobResult]`.

- [ ] **Step 1: Write failing tests `tests/test_schemas.py`**

```python
from app.schemas import EvidenceFrame, JobResult, RuleRef, Violation


def test_done_result_matches_tz_schema():
    r = JobResult(job_id="abc", status="done", verdict="violation", violations=[
        Violation(type="Наличные оставлены без присмотра", timecode_sec=9.4, description="d",
                  evidence_frames=[EvidenceFrame(timestamp_sec=9.4, image_base64="xx")],
                  rule_id="R01", rule=RuleRef(act="Правила № 231", point="п. 20", strength="В"))])
    d = r.to_api()
    assert set(d) >= {"job_id", "status", "verdict", "violations"}
    v = d["violations"][0]
    assert set(v) >= {"type", "timecode_sec", "description", "evidence_frames"}
    assert v["evidence_frames"][0] == {"timestamp_sec": 9.4, "image_base64": "xx"}


def test_processing_result_has_no_verdict_or_violations():
    d = JobResult(job_id="abc", status="processing").to_api()
    assert d == {"job_id": "abc", "status": "processing"}


def test_no_violation_keeps_empty_list():
    d = JobResult(job_id="a", status="done", verdict="no_violation").to_api()
    assert d["violations"] == []
```

- [ ] **Step 2: Run** `pytest tests/test_schemas.py -q` — Expected: FAIL (`ModuleNotFoundError: app.schemas`).

- [ ] **Step 3: Implement `app/config.py`**

```python
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str = "dev-key"
    jobs_dir: Path = Path("jobs")
    vlm_backend: str = "openai"          # "openai" | "fake"
    vlm_url: str = "http://127.0.0.1:8000/v1"
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    vlm_coord_mode: str = "rel1000"      # set from scripts/check_grounding.py
    vlm_workers: int = 16
    detector: str = "frcnn_mobile"       # "frcnn_mobile" (CPU) | "frcnn_r50v2" (GPU)
    device: str = "cpu"
    det_fps: float = 5.0
    probe_fps: float = 1.0
    max_probe_frames: int = 240
    time_budget_s: float = 200.0
    max_upload_mb: int = 300


def load_settings() -> Settings:
    e = os.environ
    return Settings(
        api_key=e.get("API_KEY", "dev-key"),
        jobs_dir=Path(e.get("JOBS_DIR", "jobs")),
        vlm_backend=e.get("VLM_BACKEND", "openai"),
        vlm_url=e.get("VLM_URL", "http://127.0.0.1:8000/v1"),
        vlm_model=e.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        vlm_coord_mode=e.get("VLM_COORD_MODE", "rel1000"),
        detector=e.get("DETECTOR", "frcnn_mobile"),
        device=e.get("DEVICE", "cpu"),
        time_budget_s=float(e.get("TIME_BUDGET_S", "200")),
    )
```

- [ ] **Step 4: Implement `app/schemas.py`**

```python
from typing import Literal

from pydantic import BaseModel


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class RuleRef(BaseModel):
    act: str
    point: str
    strength: str
    url: str | None = None


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame]
    rule_id: str | None = None
    rule: RuleRef | None = None
    confidence: float | None = None


class JobResult(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["violation", "no_violation"] | None = None
    violations: list[Violation] = []
    error: str | None = None

    def to_api(self) -> dict:
        data = self.model_dump(exclude_none=True)
        if self.status != "done":
            data.pop("violations", None)
        return data
```

- [ ] **Step 5: Run** `pytest tests/test_schemas.py -q` — Expected: `3 passed`.

- [ ] **Step 6: Write failing tests `tests/test_jobs.py`**

```python
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
```

- [ ] **Step 7: Run** `pytest tests/test_jobs.py -q` — Expected: FAIL (`ModuleNotFoundError: app.jobs`).

- [ ] **Step 8: Implement `app/jobs.py`**

```python
import queue
import threading
import traceback
import uuid
from pathlib import Path
from typing import Callable

from app.schemas import JobResult

ProcessFn = Callable[[Path, str], JobResult]


class JobStore:
    """One directory per job: input.mov + result.json (written atomically)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_job(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        (self.root / job_id).mkdir()
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
                result = self.process(self.store.video_path(job_id), job_id)
            except Exception as e:  # one bad video must not stop the worker
                traceback.print_exc()
                result = JobResult(job_id=job_id, status="failed", error=f"{type(e).__name__}: {e}")
            self.store.save(result)
            self.q.task_done()
```

- [ ] **Step 9: Run** `pytest tests/test_jobs.py -q` — Expected: `6 passed`.

- [ ] **Step 10: Checkpoint** — `pytest -q` green.

---

### Task 2: API with key auth and stub processor (Track A)

**Files:**
- Create: `app/api.py`, `app/stub.py`, `app/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Settings`, `JobStore`, `Worker`, `ProcessFn`, `JobResult` (Task 1).
- Produces: `create_app(settings, process) -> FastAPI`; module-level `app.main:app`; `app.stub.stub_process(path, job_id) -> JobResult`.

- [ ] **Step 1: Write failing tests `tests/test_api.py`**

```python
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
```

- [ ] **Step 2: Run** `pytest tests/test_api.py -q` — Expected: FAIL (`ModuleNotFoundError: app.api`).

- [ ] **Step 3: Implement `app/api.py`**

```python
import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import Settings
from app.jobs import JobStore, ProcessFn, Worker

STATIC = Path(__file__).parent / "static"
MB = 1 << 20


def create_app(settings: Settings, process: ProcessFn) -> FastAPI:
    store = JobStore(settings.jobs_dir)
    worker = Worker(store, process)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        yield

    app = FastAPI(title="Cash discipline video checker", lifespan=lifespan)

    def check_key(key: str | None) -> None:
        if not key or not hmac.compare_digest(key.encode(), settings.api_key.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.post("/api/v1/analyze")
    async def analyze(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        job_id, path = store.new_job()
        size = 0
        try:
            with open(path, "wb") as f:
                while chunk := await file.read(MB):
                    size += len(chunk)
                    if size > settings.max_upload_mb * MB:
                        raise HTTPException(413, "file too large")
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, "empty file")
        except HTTPException as e:
            store.fail(job_id, str(e.detail))
            raise
        worker.submit(job_id)
        return {"job_id": job_id, "status": "processing"}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        r = store.load(job_id)
        if r is None:
            raise HTTPException(404, "job not found")
        return JSONResponse(r.to_api())

    @app.get("/health")
    def health():
        return {"ok": True, "queued": worker.q.qsize()}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
```

- [ ] **Step 4: Implement `app/stub.py` and a placeholder UI so `/` works**

```python
import time
from pathlib import Path

from app.schemas import JobResult


def stub_process(path: Path, job_id: str) -> JobResult:
    """First-deploy stand-in until the real pipeline (Task 15) is wired in."""
    time.sleep(2)
    return JobResult(job_id=job_id, status="done", verdict="no_violation", violations=[])
```

Run: `printf '<!doctype html><title>Проверка кассы</title><p>UI в Task 3</p>' > app/static/index.html`

- [ ] **Step 5: Implement `app/main.py`**

```python
from app.api import create_app
from app.config import load_settings
from app.stub import stub_process

settings = load_settings()
app = create_app(settings, stub_process)
```

- [ ] **Step 6: Run** `pytest tests/test_api.py -q` — Expected: `6 passed`.

- [ ] **Step 7: Manual check**

Run: `API_KEY=k .venv/bin/uvicorn app.main:app --port 8080` and in another shell
`curl -s -X POST localhost:8080/api/v1/analyze -H "X-API-Key: k" -F "file=@open/videos/X-1.mov"` → `{"job_id":"…","status":"processing"}`; after 3 s `curl -s localhost:8080/api/v1/jobs/<id> -H "X-API-Key: k"` → `"status":"done"`.

- [ ] **Step 8: Checkpoint** — `pytest -q` green.

---

### Task 3: One-page UI (Track A)

**Files:**
- Modify: `app/static/index.html` (replace placeholder)

**Interfaces:**
- Consumes: `POST /api/v1/analyze`, `GET /api/v1/jobs/{id}` (Task 2); result fields from `JobResult.to_api()` including `rule` (`act`, `point`, `url`, `strength`).

- [ ] **Step 1: Write `app/static/index.html`**

```html
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка кассовой дисциплины</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#5b6475; --ok:#1a7f37; --bad:#c62828; --line:#dde1e8; }
  @media (prefers-color-scheme: dark) { :root { --bg:#14171c; --card:#1d222a; --text:#e8ebf0; --muted:#9aa3b2; --line:#2c333d; } }
  body { margin:0; font:16px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--text); }
  main { max-width:960px; margin:0 auto; padding:24px 16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:12px 0; }
  .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
  input[type=text] { flex:1; min-width:180px; padding:8px; border:1px solid var(--line); border-radius:6px; background:transparent; color:inherit; }
  button { padding:8px 16px; border:0; border-radius:6px; background:#2457d6; color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .verdict { font-size:22px; font-weight:700; }
  .ok { color:var(--ok); } .bad { color:var(--bad); }
  .frames { display:flex; gap:8px; overflow-x:auto; }
  .frames figure { margin:0; } .frames img { height:240px; border-radius:6px; display:block; }
  .frames figcaption, .muted { color:var(--muted); font-size:14px; }
  progress { width:100%; }
</style>
</head>
<body>
<main>
  <h1>Проверка видео кассира</h1>
  <p class="muted">Результат — подсказка для проверяющего, а не юридическое заключение.</p>
  <div class="card">
    <div class="row"><label for="key">API-ключ</label><input id="key" type="text" autocomplete="off"></div>
    <div class="row" style="margin-top:8px"><input id="file" type="file" accept=".mov,video/*"><button id="go">Проверить</button></div>
    <progress id="up" value="0" max="100" hidden></progress>
    <p id="status" class="muted"></p>
  </div>
  <div id="result"></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
try { $("key").value = params.get("key") || localStorage.getItem("apiKey") || ""; } catch (e) {}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

$("go").onclick = () => {
  const key = $("key").value.trim(), f = $("file").files[0];
  if (!key || !f) { $("status").textContent = "Укажите ключ и выберите файл."; return; }
  try { localStorage.setItem("apiKey", key); } catch (e) {}
  const fd = new FormData(); fd.append("file", f);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/v1/analyze"); xhr.setRequestHeader("X-API-Key", key);
  $("up").hidden = false; $("go").disabled = true; $("result").innerHTML = "";
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) $("up").value = 100 * e.loaded / e.total; };
  xhr.onload = () => {
    $("up").hidden = true;
    if (xhr.status !== 200) { $("status").textContent = "Ошибка загрузки: " + xhr.status + " " + xhr.responseText; $("go").disabled = false; return; }
    poll(JSON.parse(xhr.responseText).job_id, key, Date.now());
  };
  xhr.onerror = () => { $("status").textContent = "Сеть недоступна"; $("go").disabled = false; };
  xhr.send(fd);
};

async function poll(id, key, t0) {
  const r = await fetch("/api/v1/jobs/" + id, { headers: { "X-API-Key": key } });
  const d = await r.json();
  if (d.status === "processing") {
    $("status").textContent = "Идёт анализ… " + Math.round((Date.now() - t0) / 1000) + " с";
    setTimeout(() => poll(id, key, t0), 2000); return;
  }
  $("go").disabled = false;
  if (d.status === "failed") { $("status").textContent = "Не удалось обработать: " + (d.error || ""); return; }
  $("status").textContent = "Готово за " + Math.round((Date.now() - t0) / 1000) + " с";
  render(d);
}

function render(d) {
  const bad = d.verdict === "violation";
  let html = `<div class="card"><div class="verdict ${bad ? "bad" : "ok"}">${bad ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}</div></div>`;
  for (const v of d.violations || []) {
    const rule = v.rule ? `<p><b>Норматив:</b> ${v.rule.url ? `<a href="${esc(v.rule.url)}" target="_blank" rel="noopener">${esc(v.rule.act)}</a>` : esc(v.rule.act)}, ${esc(v.rule.point)}</p>` : "";
    const frames = v.evidence_frames.map((f) => `<figure><img alt="кадр ${f.timestamp_sec} с" src="data:image/jpeg;base64,${f.image_base64}"><figcaption>${f.timestamp_sec.toFixed(1)} с</figcaption></figure>`).join("");
    html += `<div class="card"><h3>${esc(v.type)} — ${v.timecode_sec.toFixed(1)} с</h3><p>${esc(v.description)}</p>${rule}<div class="frames">${frames}</div></div>`;
  }
  $("result").innerHTML = html;
}
</script>
</body>
</html>
```

- [ ] **Step 2: Manual check** — run uvicorn as in Task 2 Step 7, open `http://localhost:8080/?key=k`, upload `open/videos/X-1.mov`: progress bar moves, status counts seconds, then "Нарушение не обнаружено". Check the page at 375 px width (DevTools device mode): no horizontal scroll.

- [ ] **Step 3: Checkpoint** — `pytest -q` green (`test_index_served`).

---

### Task 4: GPU host, vLLM, public URL (Track A)

**Files:**
- Create: `deploy/setup.sh`, `deploy/start.sh`

**Interfaces:**
- Produces: vLLM OpenAI API at `http://127.0.0.1:8000/v1` serving `$VLM_MODEL`; app at `:8080`; public HTTPS URL via Cloudflare Tunnel.

- [ ] **Step 1: Rent the host** — Vast.ai, RTX 3090 24 GB, on-demand, reliability > 99 %, disk 80–100 GB, template "PyTorch (CUDA 12.x)". Top up $10–15. SSH in.

- [ ] **Step 2: Write `deploy/setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"

python3 -m venv /opt/vllm && /opt/vllm/bin/pip install -U pip vllm
python3 -m venv .venv && .venv/bin/pip install -U pip -r requirements.txt

# weights downloaded once; processing runs with HF_HUB_OFFLINE=1
/opt/vllm/bin/huggingface-cli download "$VLM_MODEL"
.venv/bin/python -c "import torchvision.models.detection as d; d.fasterrcnn_resnet50_fpn_v2(weights=d.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)"

curl -fsSL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
echo "setup done"
```

- [ ] **Step 3: Write `deploy/start.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
: "${API_KEY:?set API_KEY}"

/opt/vllm/bin/vllm serve "$VLM_MODEL" --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.80 --limit-mm-per-prompt '{"image": 10}' \
  --max-num-seqs 32 > vllm.log 2>&1 &

until curl -sf localhost:8000/v1/models > /dev/null; do sleep 2; done
echo "vLLM up"

DETECTOR=frcnn_r50v2 DEVICE=cuda VLM_MODEL="$VLM_MODEL" \
  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1 > app.log 2>&1 &

cloudflared tunnel --url http://localhost:8080 > tunnel.log 2>&1 &
sleep 8; grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log | head -1
```

- [ ] **Step 4: Run** `bash deploy/setup.sh && API_KEY=<generate: python3 -c "import secrets;print(secrets.token_urlsafe(24))"> bash deploy/start.sh`
Expected: prints `vLLM up` and a `https://….trycloudflare.com` URL. If the FP8 checkpoint does not exist or fails to load on Ampere, set `VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct` with `--max-model-len 4096`, or `Qwen/Qwen3-VL-4B-Instruct`; check `vllm.log`.

- [ ] **Step 5: Verify logprobs work end to end**

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model": "'"$VLM_MODEL"'", "max_tokens": 1, "temperature": 0, "logprobs": true, "top_logprobs": 5,
 "messages": [{"role": "user", "content": "Is the sky blue? Answer with exactly one word: Yes or No."}]}' | python3 -m json.tool | grep -A3 top_logprobs
```
Expected: a `top_logprobs` list containing a `"Yes"` token.

- [ ] **Step 6: Public check from another machine** — `curl -X POST https://<url>/api/v1/analyze -H "X-API-Key: <key>" -F "file=@X-1.mov"` returns a job_id; UI opens at `https://<url>/?key=<key>`.

- [ ] **Step 7: Stable URL** — the quick tunnel URL changes on restart. If the team has a domain on Cloudflare, create a named tunnel (`cloudflared tunnel login && cloudflared tunnel create kassa && cloudflared tunnel route dns kassa kassa.<domain>`), run `cloudflared tunnel run --url http://localhost:8080 kassa`. Otherwise keep the quick tunnel process alive under `nohup` and do not restart it after submitting the URL.

- [ ] **Step 8: Offline demo** — on the host, after start: `iptables -A OUTPUT -o lo -j ACCEPT; iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT; iptables -A OUTPUT -p tcp --dport 7844 -j ACCEPT; iptables -A OUTPUT -p udp --dport 7844 -j ACCEPT; iptables -A OUTPUT -j DROP` (port 7844 = Cloudflare tunnel only), then process a video via the public URL → still `done`. Undo with `iptables -F OUTPUT`. (If the container lacks iptables rights, demonstrate on a local machine with Wi-Fi off.)

---

### Task 5: Geometry and frame decoding — S0 (Track B)

**Files:**
- Create: `app/pipeline/geom.py`, `app/pipeline/frames.py`
- Test: `tests/test_geom.py`, `tests/test_frames.py`

**Interfaces:**
- Produces: `Box = tuple[float,float,float,float]` (normalized x1,y1,x2,y2); `clip(b)`, `expand(b,k)`, `center(b)`, `contains(b,x,y)`, `from_pixels(xyxy,w,h)`, `crop(img, b|None, margin=0.15)`; `VideoFrames(times, images, duration, brightness, sharpness)` with `.index_at(t)`, `.at(t)`, `.times_between(t0,t1)`; `load_frames(path, fps) -> VideoFrames`; `apply_rotation(img, degrees_ccw)`.

- [ ] **Step 1: Write failing tests `tests/test_geom.py`**

```python
import numpy as np

from app.pipeline.geom import center, clip, contains, crop, expand, from_pixels


def test_clip_and_expand():
    assert clip((-0.1, 0.2, 1.3, 0.9)) == (0.0, 0.2, 1.0, 0.9)
    assert expand((0.4, 0.4, 0.6, 0.6), 2.0) == (0.3, 0.3, 0.7, 0.7)


def test_center_contains_from_pixels():
    b = from_pixels((10, 20, 30, 60), 100, 200)
    assert b == (0.1, 0.1, 0.3, 0.3)
    assert center(b) == (0.2, 0.2) and contains(b, 0.2, 0.2) and not contains(b, 0.5, 0.5)


def test_crop_none_returns_full_and_box_returns_region():
    img = np.zeros((200, 100, 3), np.uint8)
    assert crop(img, None) is img
    c = crop(img, (0.4, 0.4, 0.6, 0.6), margin=0.0)
    assert c.shape[:2] == (40, 20)
```

- [ ] **Step 2: Run** `pytest tests/test_geom.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `app/pipeline/geom.py`**

```python
"""Normalized box helpers. A Box is (x1, y1, x2, y2) in [0, 1] relative to the frame."""
import numpy as np

Box = tuple[float, float, float, float]


def clip(b: Box) -> Box:
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in b)
    return (x1, y1, x2, y2)


def expand(b: Box, k: float) -> Box:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    hw, hh = (b[2] - b[0]) * k / 2, (b[3] - b[1]) * k / 2
    return clip((round(cx - hw, 6), round(cy - hh, 6), round(cx + hw, 6), round(cy + hh, 6)))


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def contains(b: Box, x: float, y: float) -> bool:
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def from_pixels(xyxy, w: int, h: int) -> Box:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return clip((x1 / w, y1 / h, x2 / w, y2 / h))


def crop(img: np.ndarray, b: Box | None, margin: float = 0.15) -> np.ndarray:
    if b is None:
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = expand(b, 1 + 2 * margin) if margin else b
    px1, py1 = int(round(x1 * w)), int(round(y1 * h))
    px2, py2 = max(px1 + 8, int(round(x2 * w))), max(py1 + 8, int(round(y2 * h)))
    return img[py1:min(h, py2), px1:min(w, px2)]
```

- [ ] **Step 4: Run** `pytest tests/test_geom.py -q` — Expected: `3 passed`.

- [ ] **Step 5: Write failing tests `tests/test_frames.py`**

```python
import numpy as np

from app.pipeline.frames import apply_rotation, load_frames


def ramp(i):
    return np.full((160, 96, 3), min(255, i * 3), np.uint8)


def test_samples_at_requested_fps(tmp_video):
    vf = load_frames(tmp_video(90, 30, ramp), fps=5.0)
    assert 14 <= len(vf.times) <= 16
    assert abs(vf.times[1] - vf.times[0] - 0.2) < 0.05
    assert vf.brightness[-1] > vf.brightness[0]
    assert 2.8 <= vf.duration <= 3.1


def test_at_returns_nearest_frame(tmp_video):
    vf = load_frames(tmp_video(90, 30, ramp), fps=5.0)
    assert abs(vf.at(2.0).mean() - 180) < 25      # frame ~60 -> 180
    assert vf.index_at(-1) == 0 and vf.index_at(99) == len(vf.times) - 1


def test_timestamps_start_at_zero_with_offset_stream(tmp_video):
    vf = load_frames(tmp_video(30, 30, ramp), fps=5.0)
    assert vf.times[0] < 0.05


def test_apply_rotation_clockwise():
    img = np.zeros((4, 2, 3), np.uint8)
    img[0, 0] = 255                       # marker top-left
    out = apply_rotation(img, -90)        # display matrix -90 ccw = rotate clockwise
    assert out.shape[:2] == (2, 4) and out[0, -1, 0] == 255
```

- [ ] **Step 6: Run** `pytest tests/test_frames.py -q` — Expected: FAIL (module missing).

- [ ] **Step 7: Implement `app/pipeline/frames.py`**

```python
"""S0: decode a video into upright RGB frames sampled at a fixed rate, timestamps from stream start."""
import bisect
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass
class VideoFrames:
    times: list[float]
    images: list[np.ndarray]
    duration: float
    brightness: list[float] = field(default_factory=list)
    sharpness: list[float] = field(default_factory=list)

    def index_at(self, t: float) -> int:
        i = bisect.bisect_left(self.times, t)
        if i <= 0:
            return 0
        if i >= len(self.times):
            return len(self.times) - 1
        return i if self.times[i] - t < t - self.times[i - 1] else i - 1

    def at(self, t: float) -> np.ndarray:
        return self.images[self.index_at(t)]

    def times_between(self, t0: float, t1: float) -> list[float]:
        return [t for t in self.times if t0 <= t <= t1]


def apply_rotation(img: np.ndarray, degrees_ccw: float) -> np.ndarray:
    """PyAV frame.rotation is the display-matrix angle (counter-clockwise); np.rot90 is counter-clockwise too."""
    k = int(round(degrees_ccw / 90)) % 4
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def load_frames(path: Path, fps: float) -> VideoFrames:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        start = stream.start_time or 0
        times, images, bright, sharp = [], [], [], []
        next_t = 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float((frame.pts - start) * stream.time_base)
            if t + 1e-6 < next_t:
                continue
            while next_t <= t + 1e-6:
                next_t += 1.0 / fps
            img = apply_rotation(frame.to_ndarray(format="rgb24"), getattr(frame, "rotation", 0) or 0)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            times.append(round(max(0.0, t), 3))
            images.append(img)
            bright.append(float(gray.mean()))
            sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        stream_dur = float(stream.duration * stream.time_base) if stream.duration else 0.0
        duration = max(times[-1] if times else 0.0, stream_dur)
        return VideoFrames(times, images, duration, bright, sharp)
    finally:
        container.close()
```

- [ ] **Step 8: Run** `pytest tests/test_frames.py -q` — Expected: `4 passed`.

- [ ] **Step 9: Checkpoint** — `pytest -q` green.

---

### Task 6: Person detector and tracking — S2a (Track B)

**Files:**
- Create: `app/pipeline/people.py` (detector + tracking part)
- Test: `tests/test_people_tracking.py`

**Interfaces:**
- Consumes: `VideoFrames`, `from_pixels`, `Box` (Task 5).
- Produces: `Track(id, times, boxes)` with `.start`, `.end`, `.height()`, `.box_at(t, tol=0.25) -> Box|None`; `TorchvisionPersonDetector(kind, device, score=0.5)` callable `list[np.ndarray] -> list[np.ndarray (N,5) px xyxy+score]`; `track_people(frames, detector, fps, batch=8) -> list[Track]`.

- [ ] **Step 1: Write failing tests `tests/test_people_tracking.py`**

```python
import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.people import Track, track_people


class ScriptedDetector:
    """Returns one person box per frame whose x position is encoded in the frame's red channel."""

    def __call__(self, images):
        out = []
        for im in images:
            x = int(im[0, 0, 0])
            out.append(np.array([[x, 50, x + 20, 120, 0.9]], np.float32) if x > 0 else np.zeros((0, 5), np.float32))
        return out


def frames_with_person(xs):
    imgs = []
    for x in xs:
        im = np.zeros((160, 96, 3), np.uint8)
        im[0, 0, 0] = x
        imgs.append(im)
    return VideoFrames([round(i * 0.2, 3) for i in range(len(xs))], imgs, len(xs) * 0.2)


def test_track_box_at_and_height():
    tr = Track(1, [0.0, 0.2, 0.4], [(0.1, 0.1, 0.2, 0.5)] * 3)
    assert tr.box_at(0.21) == (0.1, 0.1, 0.2, 0.5) and tr.box_at(1.0) is None
    assert abs(tr.height() - 0.4) < 1e-9 and tr.start == 0.0 and tr.end == 0.4


def test_single_person_gives_one_track():
    tracks = track_people(frames_with_person([30] * 20), ScriptedDetector(), fps=5.0)
    assert len(tracks) == 1 and len(tracks[0].times) >= 17


def test_empty_video_gives_no_tracks():
    assert track_people(frames_with_person([0] * 10), ScriptedDetector(), fps=5.0) == []
```

- [ ] **Step 2: Run** `pytest tests/test_people_tracking.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement the tracking part of `app/pipeline/people.py`**

```python
"""S2: person detection + tracking, cashier identification and person events."""
import bisect
import statistics as st
from dataclasses import dataclass, field

import numpy as np
import supervision as sv
import trackers

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import Box, center, contains, expand, from_pixels

MIN_SAMPLES = 3


@dataclass
class Track:
    id: int
    times: list[float] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.times[0]

    @property
    def end(self) -> float:
        return self.times[-1]

    def height(self) -> float:
        return st.median(b[3] - b[1] for b in self.boxes)

    def box_at(self, t: float, tol: float = 0.25) -> Box | None:
        i = bisect.bisect_left(self.times, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.times) and abs(self.times[j] - t) <= tol:
                if best is None or abs(self.times[j] - t) < abs(self.times[best] - t):
                    best = j
        return self.boxes[best] if best is not None else None


class TorchvisionPersonDetector:
    """COCO Faster R-CNN from torchvision (BSD-3). kind: 'frcnn_mobile' (CPU) or 'frcnn_r50v2' (GPU)."""

    def __init__(self, kind: str = "frcnn_mobile", device: str = "cpu", score: float = 0.5):
        import torch
        import torchvision.models.detection as tvd

        if kind == "frcnn_r50v2":
            model = tvd.fasterrcnn_resnet50_fpn_v2(weights=tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.3)
        else:
            model = tvd.fasterrcnn_mobilenet_v3_large_fpn(weights=tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, box_score_thresh=0.3)
        self.model, self.device, self.score, self.torch = model.eval().to(device), device, score, torch

    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        from torchvision.transforms.functional import to_tensor

        with self.torch.no_grad():
            outs = self.model([to_tensor(im).to(self.device) for im in images])
        result = []
        for o in outs:
            keep = (o["labels"] == 1) & (o["scores"] >= self.score)
            boxes = o["boxes"][keep].cpu().numpy()
            scores = o["scores"][keep].cpu().numpy()[:, None]
            result.append(np.concatenate([boxes, scores], axis=1).astype(np.float32))
        return result


def track_people(frames: VideoFrames, detector, fps: float, batch: int = 8) -> list[Track]:
    tracker = trackers.ByteTrackTracker(frame_rate=fps, lost_track_buffer=int(fps * 2), track_activation_threshold=0.5)
    h, w = frames.images[0].shape[:2]
    tracks: dict[int, Track] = {}
    for i in range(0, len(frames.images), batch):
        for t, d in zip(frames.times[i:i + batch], detector(frames.images[i:i + batch])):
            det = sv.Detections(xyxy=d[:, :4].reshape(-1, 4), confidence=d[:, 4], class_id=np.zeros(len(d), int))
            det = tracker.update(det)
            for tid, xyxy in zip(det.tracker_id, det.xyxy):
                if tid < 0:
                    continue
                tr = tracks.setdefault(int(tid), Track(int(tid)))
                tr.times.append(t)
                tr.boxes.append(from_pixels(xyxy, w, h))
    return [tr for tr in tracks.values() if len(tr.times) >= MIN_SAMPLES]
```

- [ ] **Step 4: Run** `pytest tests/test_people_tracking.py -q` — Expected: `3 passed`.

- [ ] **Step 5: Smoke test on a real clip (manual, ~30 s on CPU)**

Run: `.venv/bin/python -c "from app.pipeline.frames import load_frames; from app.pipeline.people import *; f=load_frames('open/videos/Х-26.mov',5); tr=track_people(f, TorchvisionPersonDetector(), 5); print([(t.id, t.start, t.end, round(t.height(),2)) for t in tr])"`
Expected: one long track with height ≈ 0.15 (cashier) and short tracks with height < 0.1 (glass reflections).

- [ ] **Step 6: Checkpoint** — `pytest -q` green.

---

### Task 7: Cashier and visitor events — S2b (Track B)

**Files:**
- Modify: `app/pipeline/people.py` (append timeline part)
- Test: `tests/test_people_events.py`

**Interfaces:**
- Consumes: `Track`, `Box`, `expand`, `center`, `contains`.
- Produces: `Event(t, kind, track_id)` with kinds `cashier_left | cashier_entered | entered | exited`; `PeopleTimeline(times, cashier_id, cashier_away, others, events, seat)` with `.away_at(t) -> bool`, `.others_at(t) -> int`; `debounce(times, raw, min_s) -> list[bool]`; `analyze_people(tracks, times, desk, safe, duration, aspect) -> PeopleTimeline`.

- [ ] **Step 1: Write failing tests `tests/test_people_events.py`**

```python
from app.pipeline.people import Track, analyze_people, debounce

T = [round(i * 0.2, 3) for i in range(100)]          # 20 s at 5 fps
SEAT = (0.4, 0.4, 0.6, 0.8)
DESK = (0.2, 0.5, 0.8, 0.9)


def track(tid, t0, t1, box=SEAT):
    ts = [t for t in T if t0 <= t <= t1]
    return Track(tid, ts, [box] * len(ts))


def moved(tid, t_sit_end, t_gone, box_far=(0.05, 0.1, 0.25, 0.5)):
    ts = [t for t in T if t <= t_gone]
    return Track(tid, ts, [SEAT if t < t_sit_end else box_far for t in ts])


def test_debounce_ignores_short_blips_and_backdates():
    times = [i * 0.2 for i in range(10)]
    raw = [False, False, True, False, False, True, True, True, True, True]
    assert debounce(times, raw, 0.6) == [False, False, False, False, False, True, True, True, True, True]


def test_cashier_leaves_room_event_at_movement_start():
    tl = analyze_people([moved(1, 9.0, 12.0)], T, DESK, None, 20.0, 0.55)
    left = [e for e in tl.events if e.kind == "cashier_left"]
    assert len(left) == 1 and abs(left[0].t - 9.0) <= 0.2
    assert tl.away_at(15.0) and not tl.away_at(5.0)


def test_trip_to_safe_is_not_departure():
    safe = (0.0, 0.05, 0.3, 0.55)
    tr = Track(1, T, [(0.05, 0.1, 0.25, 0.5) if 6.0 <= t < 10.0 else SEAT for t in T])
    tl = analyze_people([tr], T, DESK, safe, 20.0, 0.55)
    assert not [e for e in tl.events if e.kind == "cashier_left"]


def test_short_dropout_is_not_departure():
    tr = Track(1, [t for t in T if not (4.0 <= t <= 4.2)], [SEAT] * (len(T) - 2))
    tl = analyze_people([tr], T, DESK, None, 20.0, 0.55)
    assert not [e for e in tl.events if e.kind == "cashier_left"]


def test_visitor_entry_exit_and_reflection_ignored():
    cashier = track(1, 0, 20)
    client = track(2, 2.0, 6.0, box=(0.3, 0.1, 0.5, 0.4))
    reflection = track(3, 1.0, 19.0, box=(0.0, 0.2, 0.05, 0.26))
    tl = analyze_people([cashier, client, reflection], T, DESK, None, 20.0, 0.55)
    kinds = [(e.kind, e.t) for e in tl.events]
    assert ("entered", 2.0) in kinds and ("exited", 6.0) in kinds
    assert not [e for e in tl.events if e.track_id == 3]
    assert tl.others_at(4.0) == 1 and tl.others_at(10.0) == 0


def test_cashier_enters_empty_room():
    tl = analyze_people([track(1, 3.0, 20)], T, DESK, None, 20.0, 0.55)
    assert [e.kind for e in tl.events] == ["cashier_entered"]


def test_no_tracks():
    tl = analyze_people([], T, DESK, None, 20.0, 0.55)
    assert tl.cashier_id is None and tl.events == [] and tl.away_at(5.0)
```

- [ ] **Step 2: Run** `pytest tests/test_people_events.py -q` — Expected: FAIL (`ImportError: analyze_people`).

- [ ] **Step 3: Append to `app/pipeline/people.py`**

```python
H_REL = 0.5        # glass reflections are < half the cashier's height (probe 2026-10-17)
AWAY_D = 0.05      # away = centre > 5 % of frame height from the seat
AWAY_S = 0.6       # ... for at least 0.6 s
CHAIN_GAP_S = 3.0  # away runs closer than this belong to one departure
EDGE_S = 1.0       # tracks starting/ending within 1 s of the clip edge are not entries/exits


@dataclass
class Event:
    t: float
    kind: str
    track_id: int


@dataclass
class PeopleTimeline:
    times: list[float]
    cashier_id: int | None
    cashier_away: list[bool]
    others: list[int]
    events: list[Event]
    seat: Box | None = None

    def _i(self, t: float) -> int:
        i = bisect.bisect_left(self.times, t)
        return min(max(i, 0), len(self.times) - 1)

    def away_at(self, t: float) -> bool:
        return self.cashier_away[self._i(t)] if self.times else True

    def others_at(self, t: float) -> int:
        return self.others[self._i(t)] if self.times else 0


def debounce(times: list[float], raw: list[bool], min_s: float) -> list[bool]:
    """Accept a state change only if it lasts min_s; accepted changes are back-dated to their first sample."""
    out = list(raw)
    state, run = raw[0], None
    for i, v in enumerate(raw):
        if v == state:
            if run is not None:
                out[run:i] = [state] * (i - run)
                run = None
        else:
            run = i if run is None else run
            if times[i] - times[run] >= min_s:
                state, run = v, None
    if run is not None:
        out[run:] = [state] * (len(raw) - run)
    return out


def _runs(times: list[float], flags: list[bool]) -> list[tuple[int, int]]:
    runs, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def analyze_people(tracks: list[Track], times: list[float], desk: Box | None, safe: Box | None,
                   duration: float, aspect: float) -> PeopleTimeline:
    if not tracks or not times:
        return PeopleTimeline(times, None, [True] * len(times), [0] * len(times), [])
    near_desk = expand(desk, 1.4) if desk else None

    def desk_score(tr: Track) -> tuple[int, int]:
        hits = sum(contains(near_desk, *center(b)) for b in tr.boxes) if near_desk else len(tr.boxes)
        return hits, len(tr.boxes)

    cashier = max(tracks, key=desk_score)
    real = [tr for tr in tracks if tr.height() >= H_REL * cashier.height()]
    sx = st.median(center(b)[0] for b in cashier.boxes)
    sy = st.median(center(b)[1] for b in cashier.boxes)
    near_safe = expand(safe, 1.5) if safe else None

    raw = []
    for t in times:
        b = cashier.box_at(t)
        if b is None:
            raw.append(True)
            continue
        cx, cy = center(b)
        far = ((cx - sx) * aspect) ** 2 + (cy - sy) ** 2 > AWAY_D ** 2
        raw.append(far and not (near_safe and contains(near_safe, cx, cy)))
    away = debounce(times, raw, AWAY_S)

    events: list[Event] = []
    runs = _runs(times, away)
    if runs and runs[0][0] == 0 and cashier.start > EDGE_S:          # room empty at start
        events.append(Event(cashier.start, "cashier_entered", cashier.id))
        runs = runs[1:]
    chains: list[list[int]] = []                                       # merge runs with short gaps
    for a, b in runs:
        if chains and times[a] - times[chains[-1][1]] <= CHAIN_GAP_S:
            chains[-1][1] = b
        else:
            chains.append([a, b])
    for a, b in chains:
        gone_for_good = b == len(times) - 1 or cashier.end <= times[b]
        if gone_for_good and cashier.end < duration - EDGE_S:
            events.append(Event(times[a], "cashier_left", cashier.id))

    others = [sum(1 for tr in real if tr.id != cashier.id and tr.box_at(t) is not None) for t in times]
    for tr in real:
        if tr.id == cashier.id:
            continue
        if tr.start > EDGE_S:
            events.append(Event(tr.start, "entered", tr.id))
        if tr.end < duration - EDGE_S:
            events.append(Event(tr.end, "exited", tr.id))
    events.sort(key=lambda e: e.t)
    seat = (sx - 0.1, sy - 0.1, sx + 0.1, sy + 0.1)
    return PeopleTimeline(times, cashier.id, away, others, events, seat)
```

- [ ] **Step 4: Run** `pytest tests/test_people_events.py -q` — Expected: `7 passed`. If `test_cashier_enters_empty_room` yields an extra `cashier_left`, check that the entering run is removed before chaining.

- [ ] **Step 5: Checkpoint** — `pytest -q` green.

---

### Task 8: Zone change and hand activity — S3 (Track B)

**Files:**
- Create: `app/pipeline/change.py`
- Test: `tests/test_change.py`

**Interfaces:**
- Consumes: `VideoFrames`, `Track`, `Box`.
- Produces: `zone_change(frames, box, tracks, mask_people=True, diff_thr=25) -> list[tuple[float, float]]`; `change_times(series, thr) -> list[float]`; `dense_times(change_ts, step=0.2, span=0.4) -> list[float]`.

- [ ] **Step 1: Write failing tests `tests/test_change.py`**

```python
import numpy as np

from app.pipeline.change import change_times, dense_times, zone_change
from app.pipeline.frames import VideoFrames
from app.pipeline.people import Track

DESK = (0.0, 0.5, 1.0, 1.0)


def frames(square_from=None, n=10):
    imgs = []
    for i in range(n):
        im = np.zeros((100, 100, 3), np.uint8)
        if square_from is not None and i >= square_from:
            im[60:80, 40:60] = 255
        imgs.append(im)
    return VideoFrames([i * 0.2 for i in range(n)], imgs, n * 0.2)


def test_object_appearing_on_desk_spikes_once():
    s = zone_change(frames(square_from=5), DESK, [])
    spikes = change_times(s, 0.02)
    assert spikes == [1.0]


def test_change_hidden_by_person_is_masked_but_visible_as_hand_activity():
    person = Track(1, [i * 0.2 for i in range(10)], [(0.3, 0.5, 0.7, 0.9)] * 10)
    vf = frames(square_from=5)
    assert change_times(zone_change(vf, DESK, [person], mask_people=True), 0.02) == []
    assert change_times(zone_change(vf, DESK, [person], mask_people=False), 0.02) == [1.0]


def test_no_zone_returns_empty():
    assert zone_change(frames(), None, []) == []


def test_dense_times():
    assert dense_times([1.0]) == [0.6, 0.8, 1.0, 1.2, 1.4]
```

- [ ] **Step 2: Run** `pytest tests/test_change.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `app/pipeline/change.py`**

```python
"""S3: cheap per-zone change between consecutive sampled frames.

mask_people=True  -> scene changes (object put down / taken away), people ignored.
mask_people=False -> activity incl. hands (handover, counting) — the tracker is blind to these.
"""
import cv2
import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import Box, expand
from app.pipeline.people import Track


def zone_change(frames: VideoFrames, box: Box | None, tracks: list[Track],
                mask_people: bool = True, diff_thr: int = 25) -> list[tuple[float, float]]:
    if box is None or not frames.images:
        return []
    h, w = frames.images[0].shape[:2]
    x1, y1 = int(box[0] * w), int(box[1] * h)
    x2, y2 = max(x1 + 2, int(box[2] * w)), max(y1 + 2, int(box[3] * h))
    series, prev = [], None
    for t, img in zip(frames.times, frames.images):
        g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY).astype(np.int16)
        mask = np.ones(g.shape, bool)
        if mask_people:
            for tr in tracks:
                b = tr.box_at(t)
                if b is None:
                    continue
                bx1, by1, bx2, by2 = expand(b, 1.1)
                mx1, my1 = max(0, int(bx1 * w) - x1), max(0, int(by1 * h) - y1)
                mx2, my2 = min(g.shape[1], int(bx2 * w) - x1), min(g.shape[0], int(by2 * h) - y1)
                if mx2 > mx1 and my2 > my1:
                    mask[my1:my2, mx1:mx2] = False
        score = 0.0
        if prev is not None and mask.any():
            score = float((np.abs(g - prev) > diff_thr)[mask].mean())
        series.append((t, score))
        prev = g
    return series


def change_times(series: list[tuple[float, float]], thr: float) -> list[float]:
    return [t for t, s in series if s > thr]


def dense_times(change_ts: list[float], step: float = 0.2, span: float = 0.4) -> list[float]:
    n = int(round(span / step))
    return sorted({round(t + k * step, 2) for t in change_ts for k in range(-n, n + 1) if t + k * step >= 0})
```

- [ ] **Step 4: Run** `pytest tests/test_change.py -q` — Expected: `4 passed`.

- [ ] **Step 5: Checkpoint** — `pytest -q` green.

---

### Task 9: VLM client — OpenAI-compatible + fake (Track C)

**Files:**
- Create: `app/pipeline/vlm.py`
- Test: `tests/test_vlm.py`

**Interfaces:**
- Produces: `VLMError`; protocol `VLM` with `yes_prob(images, question) -> float`, `yes_probs(items: list[tuple[list[np.ndarray], str]]) -> list[float]`, `text(images, prompt, max_tokens=256) -> str`; `OpenAIVLM(base_url, model, workers=16, timeout=60.0, retries=2, backoff=0.5, client=None)`; `FakeVLM(answer=..., reply=...)` with `.calls`; `p_yes(top_logprobs) -> float`; `to_data_url(img, max_side=896) -> str`.

- [ ] **Step 1: Write failing tests `tests/test_vlm.py`**

```python
import json

import httpx
import numpy as np
import pytest

from app.pipeline.vlm import FakeVLM, OpenAIVLM, VLMError, p_yes

IMG = np.zeros((20, 10, 3), np.uint8)


def response(top):
    return {"choices": [{"message": {"content": top[0]["token"]},
                         "logprobs": {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "top_logprobs": top}]}}]}


def test_p_yes_normalizes_yes_and_no():
    assert abs(p_yes([{"token": "Yes", "logprob": -0.1}, {"token": " no", "logprob": -2.4}]) - 0.909) < 0.01
    assert p_yes([{"token": "Maybe", "logprob": -0.1}]) == 0.5


def test_yes_prob_sends_logprob_request():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=response([{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.4}]))

    vlm = OpenAIVLM("http://x/v1", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    p = vlm.yes_prob([IMG], "Is it?")
    assert 0.89 < p < 0.92
    assert seen["max_tokens"] == 1 and seen["logprobs"] is True and seen["temperature"] == 0.0
    content = seen["messages"][0]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "Yes or No" in content[-1]["text"]


def test_retries_then_raises():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    vlm = OpenAIVLM("http://x/v1", "m", retries=2, backoff=0.0, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(VLMError):
        vlm.yes_prob([IMG], "Q")
    assert calls["n"] == 3


def test_yes_probs_keeps_order():
    def handler(request):
        q = json.loads(request.content)["messages"][0]["content"][-1]["text"]
        tok = "Yes" if q.startswith("A") else "No"
        return httpx.Response(200, json=response([{"token": tok, "logprob": -0.01}]))

    vlm = OpenAIVLM("http://x/v1", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert vlm.yes_probs([([IMG], "A?"), ([IMG], "B?"), ([IMG], "A2?")]) == [1.0, 0.0, 1.0]


def test_fake_vlm():
    f = FakeVLM(answer=lambda imgs, q: 0.8 if "safe" in q else 0.1, reply=lambda imgs, p: '{"bbox_2d": null}')
    assert f.yes_probs([([IMG], "safe open?"), ([IMG], "door?")]) == [0.8, 0.1]
    assert f.text([IMG], "x") == '{"bbox_2d": null}' and f.calls == 3
```

- [ ] **Step 2: Run** `pytest tests/test_vlm.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `app/pipeline/vlm.py`**

```python
"""VLM access. OpenAIVLM talks to a local vLLM OpenAI-compatible server; FakeVLM is for tests."""
import base64
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol

import httpx
import numpy as np
from PIL import Image

YES_NO_SUFFIX = "\nAnswer with exactly one word: Yes or No."


class VLMError(RuntimeError):
    pass


class VLM(Protocol):
    def yes_prob(self, images: list[np.ndarray], question: str) -> float: ...
    def yes_probs(self, items: list[tuple[list[np.ndarray], str]]) -> list[float]: ...
    def text(self, images: list[np.ndarray], prompt: str, max_tokens: int = 256) -> str: ...


def to_data_url(img: np.ndarray, max_side: int = 896) -> str:
    im = Image.fromarray(img)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def p_yes(top_logprobs: list[dict]) -> float:
    py = pn = 0.0
    for item in top_logprobs:
        tok = item["token"].strip().strip(".").lower()
        if tok == "yes":
            py += math.exp(item["logprob"])
        elif tok == "no":
            pn += math.exp(item["logprob"])
    return 0.5 if py + pn == 0 else py / (py + pn)


class OpenAIVLM:
    def __init__(self, base_url: str, model: str, workers: int = 16, timeout: float = 60.0,
                 retries: int = 2, backoff: float = 0.5, client: httpx.Client | None = None):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.workers, self.retries, self.backoff = workers, retries, backoff
        self.client = client or httpx.Client(timeout=timeout)

    def _chat(self, images: list[np.ndarray], text: str, max_tokens: int, logprobs: bool) -> dict:
        content = [{"type": "image_url", "image_url": {"url": to_data_url(im)}} for im in images]
        content.append({"type": "text", "text": text})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": max_tokens, "temperature": 0.0}
        if logprobs:
            payload.update(logprobs=True, top_logprobs=10)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.client.post(f"{self.base_url}/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                last = e
                time.sleep(self.backoff * (attempt + 1))
        raise VLMError(f"VLM unavailable after {self.retries + 1} attempts: {last}")

    def yes_prob(self, images: list[np.ndarray], question: str) -> float:
        choice = self._chat(images, question + YES_NO_SUFFIX, 1, True)["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        if not content:
            return 1.0 if choice["message"]["content"].strip().lower().startswith("yes") else 0.0
        return p_yes(content[0]["top_logprobs"])

    def yes_probs(self, items: list[tuple[list[np.ndarray], str]]) -> list[float]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda it: self.yes_prob(*it), items))

    def text(self, images: list[np.ndarray], prompt: str, max_tokens: int = 256) -> str:
        return self._chat(images, prompt, max_tokens, False)["choices"][0]["message"]["content"]


class FakeVLM:
    def __init__(self, answer: Callable[[list[np.ndarray], str], float] = lambda i, q: 0.0,
                 reply: Callable[[list[np.ndarray], str], str] = lambda i, p: ""):
        self.answer, self.reply, self.calls = answer, reply, 0

    def yes_prob(self, images, question):
        self.calls += 1
        return float(self.answer(images, question))

    def yes_probs(self, items):
        return [self.yes_prob(i, q) for i, q in items]

    def text(self, images, prompt, max_tokens=256):
        self.calls += 1
        return self.reply(images, prompt)
```

- [ ] **Step 4: Run** `pytest tests/test_vlm.py -q` — Expected: `5 passed`.

- [ ] **Step 5: Checkpoint** — `pytest -q` green.

---

### Task 10: Scene zones via VLM grounding — S1 (Track C)

**Files:**
- Create: `app/pipeline/scene.py`, `scripts/check_grounding.py`
- Test: `tests/test_scene.py`

**Interfaces:**
- Consumes: `VLM`, `VLMError` (Task 9); `Box`, `clip` (Task 5).
- Produces: `Zones(desk, safe, door, laptop)` with `.get(name) -> Box|None` (`"full"` → `None`); `parse_bbox(text, width, height, mode) -> Box|None`; `locate_zones(image, vlm, mode="rel1000") -> Zones`; `DEFAULT_DESK`.

- [ ] **Step 1: Write failing tests `tests/test_scene.py`**

```python
import numpy as np

from app.pipeline.scene import DEFAULT_DESK, Zones, locate_zones, parse_bbox
from app.pipeline.vlm import FakeVLM, VLMError

IMG = np.zeros((848, 464, 3), np.uint8)


def test_parse_rel1000_and_abs():
    assert parse_bbox('{"bbox_2d": [100, 200, 500, 600]}', 464, 848, "rel1000") == (0.1, 0.2, 0.5, 0.6)
    assert parse_bbox('{"bbox_2d": [46.4, 84.8, 232, 424]}', 464, 848, "abs") == (0.1, 0.1, 0.5, 0.5)


def test_parse_rejects_null_and_tiny():
    assert parse_bbox('{"bbox_2d": null}', 464, 848, "rel1000") is None
    assert parse_bbox('[10, 10, 12, 12]', 464, 848, "rel1000") is None
    assert parse_bbox("no box here", 464, 848, "rel1000") is None


def test_locate_zones_uses_vlm_and_defaults_desk():
    def reply(images, prompt):
        return '{"bbox_2d": [600, 300, 900, 600]}' if "safe" in prompt else '{"bbox_2d": null}'
    z = locate_zones(IMG, FakeVLM(reply=reply))
    assert z.safe == (0.6, 0.3, 0.9, 0.6) and z.door is None and z.desk == DEFAULT_DESK
    assert z.get("full") is None and z.get("safe") == z.safe


def test_locate_zones_survives_vlm_error():
    def reply(images, prompt):
        raise VLMError("down")
    z = locate_zones(IMG, FakeVLM(reply=reply))
    assert z == Zones(desk=DEFAULT_DESK)
```

- [ ] **Step 2: Run** `pytest tests/test_scene.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `app/pipeline/scene.py`**

```python
"""S1: locate desk, safe, door and laptop on this video's own frame (no hard-coded zones)."""
import re
from dataclasses import dataclass

import numpy as np

from app.pipeline.geom import Box, clip
from app.pipeline.vlm import VLM, VLMError

OBJECTS = {
    "desk": "the cashier's work desk (table with a laptop and a banknote counting machine)",
    "safe": "the metal safe (small grey cabinet with a door)",
    "door": "the entrance door of the room",
    "laptop": "the laptop computer screen",
}
PROMPT = ('Locate {desc} in the image. Reply with JSON only, like {{"bbox_2d": [x1, y1, x2, y2]}}. '
          'If it is not visible, reply {{"bbox_2d": null}}.')
DEFAULT_DESK: Box = (0.0, 0.3, 1.0, 0.8)
_NUM4 = re.compile(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")


@dataclass
class Zones:
    desk: Box | None = None
    safe: Box | None = None
    door: Box | None = None
    laptop: Box | None = None

    def get(self, name: str) -> Box | None:
        return None if name == "full" else getattr(self, name, None)


def parse_bbox(text: str, width: int, height: int, mode: str) -> Box | None:
    m = _NUM4.search(text or "")
    if not m:
        return None
    x1, y1, x2, y2 = (float(v) for v in m.groups())
    sx, sy = (1000.0, 1000.0) if mode == "rel1000" else (float(width), float(height))
    b = clip((round(x1 / sx, 4), round(y1 / sy, 4), round(x2 / sx, 4), round(y2 / sy, 4)))
    if b[2] - b[0] < 0.02 or b[3] - b[1] < 0.02:
        return None
    return b


def locate_zones(image: np.ndarray, vlm: VLM, mode: str = "rel1000") -> Zones:
    h, w = image.shape[:2]
    zones = Zones()
    for name, desc in OBJECTS.items():
        try:
            setattr(zones, name, parse_bbox(vlm.text([image], PROMPT.format(desc=desc), 64), w, h, mode))
        except VLMError:
            setattr(zones, name, None)
    if zones.desk is None:
        zones.desk = DEFAULT_DESK
    return zones
```

- [ ] **Step 4: Run** `pytest tests/test_scene.py -q` — Expected: `4 passed`.

- [ ] **Step 5: Write `scripts/check_grounding.py` (run on the GPU host once)**

```python
"""Find which bbox convention the served VLM uses: draws a red box at a known place and compares."""
import sys

import numpy as np

from app.pipeline.scene import parse_bbox
from app.pipeline.vlm import OpenAIVLM

url, model = sys.argv[1], sys.argv[2]
img = np.full((848, 464, 3), 200, np.uint8)
img[424:636, 232:406] = (220, 30, 30)                       # truth (0.5, 0.5, 0.875, 0.75)
truth = (0.5, 0.5, 0.875, 0.75)
text = OpenAIVLM(url, model).text([img], 'Locate the red rectangle. Reply with JSON only, like {"bbox_2d": [x1, y1, x2, y2]}.', 64)
print("raw:", text)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


for mode in ("rel1000", "abs"):
    b = parse_bbox(text, 464, 848, mode)
    print(mode, b, "IoU", round(iou(b, truth), 2) if b else None)
```

Run on host: `.venv/bin/python -m scripts.check_grounding http://127.0.0.1:8000/v1 "$VLM_MODEL"`. Set `VLM_COORD_MODE` to the mode with IoU > 0.5 in the environment used by `start.sh`.

- [ ] **Step 6: Checkpoint** — `pytest -q` green.

---

### Task 11: Fact probes — S4 (Track C)

**Files:**
- Create: `app/facts.yaml`, `app/pipeline/probes.py`
- Test: `tests/test_probes.py`

**Interfaces:**
- Consumes: `VideoFrames` (Task 5), `crop` (Task 5), `Zones` (Task 10), `VLM` (Task 9).
- Produces: `Fact(id, zone, question)`; `load_facts(path) -> list[Fact]`; `probe_times(duration, fps, extra, cap) -> list[float]`; `run_probes(frames, zones, vlm, facts, times) -> dict[str, list[tuple[float, float]]]`.

- [ ] **Step 1: Create `app/facts.yaml`**

```yaml
- id: F_cash_desk
  zone: desk
  question: "Are there banknotes (paper money) lying on the desk?"
- id: F_bag_desk
  zone: desk
  question: "Is there a cash collection bag or sack lying on the desk?"
- id: F_docs_desk
  zone: desk
  question: "Is there a journal, logbook, stamp or seal lying on the desk?"
- id: F_handover
  zone: desk
  question: "Is money or a document being passed from one person's hand to another person's hand?"
- id: F_cup
  zone: full
  question: "Is there a cup, mug, glass or bottle on the desk or in a person's hand?"
- id: F_items
  zone: full
  question: "Is there a handbag, backpack, coat or jacket lying on the desk?"
- id: F_phone
  zone: full
  question: "Is the cashier holding or looking at a mobile phone?"
- id: F_safe_open
  zone: safe
  question: "Is the door of the safe open?"
- id: F_key
  zone: safe
  question: "Is there a key inserted in the lock of the safe?"
- id: F_door_open
  zone: door
  question: "Is the entrance door of the room open?"
- id: F_screen_on
  zone: laptop
  question: "Is the laptop screen turned on and showing a document, table or program (not a lock screen or a black screen)?"
```

- [ ] **Step 2: Write failing tests `tests/test_probes.py`**

```python
from pathlib import Path

import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.probes import Fact, load_facts, probe_times, run_probes
from app.pipeline.scene import Zones
from app.pipeline.vlm import FakeVLM


def test_facts_yaml_loads():
    facts = load_facts(Path("app/facts.yaml"))
    ids = {f.id for f in facts}
    assert {"F_cash_desk", "F_safe_open", "F_door_open", "F_screen_on", "F_handover"} <= ids
    assert all(f.zone in {"desk", "safe", "door", "laptop", "full"} for f in facts)


def test_probe_times_base_plus_extra_and_cap():
    assert probe_times(3.0, 1.0, [1.4, 1.4, 9.0], cap=100) == [0.0, 1.0, 1.4, 2.0, 3.0]
    ts = probe_times(10.0, 1.0, [i / 10 for i in range(100)], cap=15)
    assert len(ts) == 15 and ts[0] == 0.0 and 10.0 in ts


def test_probe_times_long_video_subsamples_base():
    assert len(probe_times(500.0, 1.0, [], cap=100)) <= 100


def test_run_probes_series_follow_frames():
    imgs = [np.full((100, 60, 3), 255 if i >= 5 else 0, np.uint8) for i in range(10)]
    vf = VideoFrames([float(i) for i in range(10)], imgs, 9.0)
    facts = [Fact("F_cash_desk", "desk", "banknotes?"), Fact("F_door_open", "door", "door?")]
    vlm = FakeVLM(answer=lambda ims, q: 0.9 if ("banknotes" in q and ims[0].mean() > 100) else 0.1)
    s = run_probes(vf, Zones(desk=(0, 0.5, 1, 1)), vlm, facts, [0.0, 4.0, 5.0, 9.0])
    assert [p for _, p in s["F_cash_desk"]] == [0.1, 0.1, 0.9, 0.9]
    assert [p for _, p in s["F_door_open"]] == [0.1] * 4
```

- [ ] **Step 3: Run** `pytest tests/test_probes.py -q` — Expected: FAIL (module missing).

- [ ] **Step 4: Implement `app/pipeline/probes.py`**

```python
"""S4: narrow yes/no fact questions on zone crops -> P(yes) time series per fact."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import crop
from app.pipeline.scene import Zones
from app.pipeline.vlm import VLM

Series = list[tuple[float, float]]


@dataclass(frozen=True)
class Fact:
    id: str
    zone: str
    question: str


def load_facts(path: Path) -> list[Fact]:
    return [Fact(**d) for d in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


def probe_times(duration: float, fps: float, extra: list[float], cap: int) -> list[float]:
    base = [round(i / fps, 2) for i in range(int(duration * fps) + 1)]
    if len(base) > cap:
        stride = -(-len(base) // cap)
        base = base[::stride]
    budget = cap - len(base)
    extra_r = sorted({round(t, 2) for t in extra if 0 <= t <= duration} - set(base))
    if budget <= 0:
        extra_r = []
    elif len(extra_r) > budget:
        extra_r = [extra_r[int(i * len(extra_r) / budget)] for i in range(budget)]
    return sorted(base + extra_r)


def run_probes(frames: VideoFrames, zones: Zones, vlm: VLM, facts: list[Fact], times: list[float]) -> dict[str, Series]:
    items, keys = [], []
    for t in times:
        img = frames.at(t)
        for f in facts:
            items.append(([crop(img, zones.get(f.zone))], f.question))
            keys.append((f.id, t))
    series: dict[str, Series] = {f.id: [] for f in facts}
    for (fid, t), p in zip(keys, vlm.yes_probs(items)):
        series[fid].append((t, p))
    return series
```

- [ ] **Step 5: Run** `pytest tests/test_probes.py -q` — Expected: `4 passed`.

- [ ] **Step 6: Checkpoint** — `pytest -q` green.

---

### Task 12: State timeline — S5 (Track C)

**Files:**
- Create: `app/pipeline/timeline.py`
- Test: `tests/test_timeline.py`

**Interfaces:**
- Produces: `Series = list[tuple[float, float]]`; `Interval(start, end, peak_t, peak_p, strong)` with `.length`; `smooth(series, k=3)`; `intervals(series, on=0.6, off=0.4, strong_p=0.8) -> list[Interval]`; `active(ints, t, tol=0.0) -> Interval|None`; `peak_in(series, t0, t1) -> tuple[float, float]`.

- [ ] **Step 1: Write failing tests `tests/test_timeline.py`**

```python
from app.pipeline.timeline import active, intervals, peak_in, smooth


def s(ps):
    return [(float(i), p) for i, p in enumerate(ps)]


def test_step_gives_one_interval_with_backdated_start():
    iv = intervals(s([0.1, 0.1, 0.9, 0.95, 0.9, 0.1, 0.1]))
    assert len(iv) == 1 and iv[0].start == 2.0 and iv[0].end == 4.0
    assert iv[0].peak_t == 3.0 and iv[0].strong == 3


def test_single_spike_removed_by_median():
    assert intervals(s([0.1, 0.1, 0.95, 0.1, 0.1])) == []


def test_hysteresis_keeps_interval_through_middle_values():
    iv = intervals(s([0.1, 0.9, 0.9, 0.5, 0.5, 0.9, 0.9, 0.1, 0.1]))
    assert len(iv) == 1 and iv[0].start == 1.0 and iv[0].end == 6.0


def test_active_and_peak_in():
    iv = intervals(s([0.1, 0.9, 0.9, 0.9, 0.1]))
    assert active(iv, 2.0) is iv[0] and active(iv, 4.5) is None and active(iv, 4.5, tol=1.0) is iv[0]
    assert peak_in(s([0.1, 0.3, 0.7, 0.2]), 0.0, 2.0) == (2.0, 0.7)
    assert peak_in([], 1.0, 2.0) == (1.0, 0.0)


def test_smooth_short_series_unchanged():
    assert smooth([(0.0, 0.9)]) == [(0.0, 0.9)]
```

- [ ] **Step 2: Run** `pytest tests/test_timeline.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `app/pipeline/timeline.py`**

```python
"""S5: noisy P(yes) series -> state intervals (median smoothing + hysteresis)."""
import statistics as st
from dataclasses import dataclass

Series = list[tuple[float, float]]


@dataclass
class Interval:
    start: float
    end: float
    peak_t: float
    peak_p: float
    strong: int          # raw samples with p >= strong_p

    @property
    def length(self) -> float:
        return self.end - self.start


def smooth(series: Series, k: int = 3) -> Series:
    if len(series) < k:
        return list(series)
    half = k // 2
    return [(t, st.median(p for _, p in series[max(0, i - half): i + half + 1])) for i, (t, _) in enumerate(series)]


def intervals(series: Series, on: float = 0.6, off: float = 0.4, strong_p: float = 0.8) -> list[Interval]:
    raw = dict(series)
    result: list[Interval] = []
    cur: Interval | None = None
    for t, p in smooth(series):
        if cur is None:
            if p < on:
                continue
            cur = Interval(t, t, t, raw[t], 0)
        elif p < off:
            result.append(cur)
            cur = None
            continue
        cur.end = t
        if raw[t] > cur.peak_p:
            cur.peak_t, cur.peak_p = t, raw[t]
        if raw[t] >= strong_p:
            cur.strong += 1
    if cur is not None:
        result.append(cur)
    return result


def active(ints: list[Interval], t: float, tol: float = 0.0) -> Interval | None:
    for iv in ints:
        if iv.start - tol <= t <= iv.end + tol:
            return iv
    return None


def peak_in(series: Series, t0: float, t1: float) -> tuple[float, float]:
    pts = [(t, p) for t, p in series if t0 <= t <= t1]
    return max(pts, key=lambda x: x[1]) if pts else (t0, 0.0)
```

- [ ] **Step 4: Run** `pytest tests/test_timeline.py -q` — Expected: `5 passed`.

- [ ] **Step 5: Checkpoint** — `pytest -q` green.

---

### Task 13: Rule catalog and engine, tier 1 — S6 (Track C)

**Files:**
- Create: `app/rules.yaml`, `app/pipeline/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: `PeopleTimeline`, `Event` (Task 7); `Interval`, `Series`, `active`, `peak_in` (Task 12).
- Produces: `Rule(id, type, families, facts, params, norm, strength, template, verify, enabled)`; `Candidate(rule_id, t, evidence_t, fact, score, strong)`; `RuleContext(people, states, series, quality, duration)` where `quality: list[tuple[float, float, float]]` = (t, brightness, sharpness); `load_rules(path) -> list[Rule]`; `evaluate(rules, ctx) -> list[Candidate]`; `FAMILIES: dict[str, Callable[[Rule, RuleContext], list[Candidate]]]`.

- [ ] **Step 1: Create `app/rules.yaml`** (norm texts from `docs/normative_research.md`)

```yaml
# family: how the rule is triggered (see app/pipeline/rules.py FAMILIES)
# strength: В — clause directly describes what is seen; С — needs context; Н — indirect, phrase as "возможное несоответствие"
# verify: questions for S7; window = seconds relative to the candidate time; expect = yes|no; zone defaults to the fact's zone
- id: R01
  type: "Наличные или ценности оставлены без присмотра"
  family: on_cashier_left
  facts: [F_cash_desk, F_bag_desk]
  params: {tol: 1.5, strong_k: 3}
  strength: "В"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 20 ч. 2", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "Все ранее принятые наличные деньги хранятся в специальных устройствах … закрывающихся на ключ"}
  template: "В {t} с кассир покинул рабочее место; на столе остаются наличные или инкассаторская сумка (кадр {te} с). Наличные должны храниться в запирающемся устройстве на рабочем месте."
  verify:
    - {q: "Are banknotes or a cash bag lying on the desk?", window: [0, 5]}
    - {q: "Is money left on the table surface?", window: [0, 5]}
    - {q: "Is the cashier's chair empty while money remains on the desk?", window: [1, 6], zone: full}
- id: R02
  type: "Сейф оставлен открытым без присмотра"
  family: [on_cashier_left, unattended_state]
  facts: [F_safe_open]
  params: {tol: 1.5, min_s: 3.0, strong_k: 3}
  strength: "В"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 20 ч. 2, п. 80", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "специальные устройства … закрывающиеся на ключ; порядок хранения ключей … исключает несанкционированное использование"}
  template: "В {t} с дверца сейфа открыта, а кассира рядом нет (кадр {te} с)."
  verify:
    - {q: "Is the door of the safe open?", window: [0, 5]}
    - {q: "Is the safe left open?", window: [0, 5]}
    - {q: "Is nobody standing next to the open safe?", window: [0, 5], zone: full}
- id: R03
  type: "Ключ оставлен в замке сейфа"
  family: on_cashier_left
  facts: [F_key]
  params: {tol: 2.0, strong_k: 3}
  strength: "С"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 80", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "порядок хранения … ключей от специальных устройств, находящихся на рабочем месте … исключает их несанкционированное использование"}
  template: "В {t} с кассир покинул рабочее место; в замке сейфа остаётся ключ (кадр {te} с)."
  verify:
    - {q: "Is a key inserted in the lock of the safe?", window: [0, 5]}
    - {q: "Is there a key sticking out of the safe door?", window: [0, 5]}
    - {q: "Is a key left in the safe while nobody is near it?", window: [1, 6], zone: full}
- id: R04
  type: "Рабочая станция не заблокирована при уходе"
  family: on_cashier_left
  facts: [F_screen_on]
  params: {tol: 2.0, strong_k: 3}
  strength: "Н"
  norm: {act: "Требования к обеспечению информационной безопасности банков (пост. АРРФР от 03.04.2026 № 53)", point: "п. 53; п. 71 пп. 9)", url: "https://adilet.zan.kz/rus/docs/V2600038336",
         quote: "меры защиты рабочих станций и ноутбуков; блокировка рабочей станции при неактивности"}
  template: "В {t} с кассир покинул рабочее место; экран ноутбука остаётся включённым с рабочим содержимым (кадр {te} с)."
  verify:
    - {q: "Is the laptop screen on and showing a document or a table?", window: [1, 6]}
    - {q: "Is the computer unlocked with content visible on the screen?", window: [1, 6]}
    - {q: "Is the screen bright with an open program?", window: [1, 6]}
- id: R05
  type: "Кассир покинул рабочее место до окончания обслуживания клиента"
  family: cashier_left_with_client
  facts: []
  params: {client_stays_s: 2.0}
  strength: "В"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 22", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента"}
  template: "В {t} с кассир покинул рабочее место, клиент при этом остаётся у стола."
  verify:
    - {q: "Is a visitor still at the desk while the cashier's seat is empty?", window: [1, 5], zone: full}
    - {q: "Has the cashier walked away while a client is still present?", window: [0, 5], zone: full}
    - {q: "Is a client waiting alone at the cash desk?", window: [1, 5], zone: full}
- id: R06
  type: "Наличные на виду при входе другого лица"
  family: on_entry
  facts: [F_cash_desk]
  params: {before_s: 1.5, strong_k: 3}
  strength: "В"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 20 ч. 2", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "На рабочем месте кассового работника находятся наличные деньги клиента"}
  template: "В {t} с в помещение входит человек, а на столе открыто лежат наличные предыдущей операции (кадр {te} с)."
  verify:
    - {q: "Are banknotes lying openly on the desk?", window: [-2, 2]}
    - {q: "Is cash visible on the table?", window: [-2, 2]}
    - {q: "Is money left on the desk while a new person comes in?", window: [-1, 3], zone: full}
- id: R08
  type: "Дверь кассы оставлена открытой"
  family: duration
  facts: [F_door_open]
  params: {min_s: 8.0, strong_k: 4}
  strength: "Н"
  norm: {act: "Правила организации охраны и устройства помещений БВУ (пост. НБРК от 24.02.2020 № 14)", point: "п. 16 пп. 1), п. 8", url: "https://adilet.zan.kz/rus/docs/V2000020075",
         quote: "каждая касса имеет отдельный вход с закрывающейся на замок дверью"}
  template: "С {t} с дверь в кассовое помещение остаётся открытой (кадр {te} с); назначение требования о закрывающейся двери не обеспечивается."
  verify:
    - {q: "Is the room door open?", window: [0, 8]}
    - {q: "Is the entrance door left open?", window: [0, 8]}
    - {q: "Can you see through an open doorway?", window: [0, 8]}
- id: R09
  type: "Постороннее лицо в кассовом помещении"
  family: entry_during_service
  facts: []
  params: {min_stay_s: 1.0}
  strength: "Н"
  norm: {act: "Правила организации охраны и устройства помещений БВУ (пост. НБРК от 24.02.2020 № 14)", point: "п. 8; Правила № 231 п. 20 ч. 2", url: "https://adilet.zan.kz/rus/docs/V2000020075",
         quote: "кассовый узел изолирован … один закрывающийся служебный вход"}
  template: "В {t} с в помещение входит ещё один человек, пока идёт обслуживание другого посетителя."
  verify:
    - {q: "Are there two or more visitors (not the cashier) in the room at the same time?", window: [0, 4], zone: full}
    - {q: "Did another person enter while a client was being served?", window: [-1, 3], zone: full}
    - {q: "Is a second visitor standing in the room?", window: [0, 4], zone: full}
- id: R10
  type: "Личные вещи на рабочем месте кассира"
  family: presence
  facts: [F_items]
  params: {min_s: 2.0, strong_k: 3}
  strength: "В"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 12 ч. 1", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "Хранение личных вещей … работников банка осуществляется вне кассового узла"}
  template: "С {t} с на рабочем месте кассира находятся личные вещи (сумка, верхняя одежда) (кадр {te} с)."
  verify:
    - {q: "Is a handbag, backpack, coat or jacket lying on the desk?", window: [0, 4]}
    - {q: "Are personal belongings placed on the work desk?", window: [0, 4]}
    - {q: "Is there clothing or a bag on the table?", window: [0, 4]}
- id: R11
  type: "Мобильный телефон в кассовом помещении"
  family: presence
  facts: [F_phone]
  params: {min_s: 2.0, strong_k: 3}
  strength: "С"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 12 ч. 2", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "Допускается внос … средств связи, не имеющих функции фото и видео фиксации"}
  template: "С {t} с кассир пользуется мобильным телефоном (кадр {te} с); допускаются только средства связи без функции фото- и видеофиксации."
  verify:
    - {q: "Is the cashier holding a mobile phone?", window: [0, 4]}
    - {q: "Is the cashier looking at a smartphone?", window: [0, 4]}
    - {q: "Is there a phone in the cashier's hand?", window: [0, 4]}
- id: R12
  type: "Еда или напитки на рабочем месте"
  family: presence
  facts: [F_cup]
  params: {min_s: 2.0, strong_k: 3}
  strength: "Н"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 12 (перечень разрешённых к вносу предметов)", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "Допускается внос … лекарственных средств, медицинских изделий и средств связи …"}
  template: "С {t} с на рабочем месте кассира находится напиток или еда (кадр {te} с); такие предметы не входят в перечень разрешённых к вносу в кассовый узел."
  verify:
    - {q: "Is the cashier drinking or holding a cup or bottle?", window: [0, 4]}
    - {q: "Is there a cup, mug or bottle on the desk?", window: [0, 4]}
    - {q: "Is there a drink at the workplace?", window: [0, 4]}
- id: R17
  type: "Печати или журнал оставлены без присмотра"
  family: on_cashier_left
  facts: [F_docs_desk]
  params: {tol: 1.5, strong_k: 3}
  strength: "С"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 80", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "порядок хранения … печатей, … клише, именных штампов … исключает их несанкционированное использование"}
  template: "В {t} с кассир покинул рабочее место; на столе остаются печати, штампы или журнал (кадр {te} с)."
  verify:
    - {q: "Is a stamp, seal or journal left on the desk?", window: [0, 5]}
    - {q: "Are stamps or a logbook lying on the table?", window: [0, 5]}
    - {q: "Is a stamp left unattended on the desk?", window: [1, 6]}
- id: R18
  type: "Обзор камеры нарушен"
  family: quality
  facts: []
  params: {min_s: 2.0, dark: 25.0, flat: 10.0}
  strength: "Н"
  norm: {act: "Правила организации охраны и устройства помещений БВУ (пост. НБРК от 24.02.2020 № 14)", point: "п. 16 пп. 6)", url: "https://adilet.zan.kz/rus/docs/V2000020075",
         quote: "касса оборудуется видеокамерами"}
  template: "С {t} с изображение с камеры тёмное или перекрыто — контроль операций по видео невозможен."
  verify: []
```

- [ ] **Step 2: Write failing tests `tests/test_rules.py`**

```python
from pathlib import Path

from app.pipeline.people import Event, PeopleTimeline
from app.pipeline.rules import FAMILIES, RuleContext, evaluate, load_rules
from app.pipeline.timeline import Interval

T = [round(i * 0.2, 3) for i in range(150)]        # 30 s
RULES = {r.id: r for r in load_rules(Path("app/rules.yaml"))}


def people(events, away_from=None, others=None):
    away = [away_from is not None and t >= away_from for t in T]
    oth = [others(t) if others else 0 for t in T]
    return PeopleTimeline(T, 1, away, oth, events)


def ctx(p, states=None, quality=None):
    return RuleContext(p, states or {}, {}, quality or [(t, 120.0, 50.0) for t in T], 30.0)


def iv(a, b, peak=0.9, strong=5):
    return Interval(a, b, (a + b) / 2, peak, strong)


def test_catalog_is_complete():
    for r in RULES.values():
        assert r.type and r.norm["act"] and r.norm["point"] and r.strength in {"В", "С", "Н"}
        assert all(f in FAMILIES for f in r.families), r.id
        assert "{t}" in r.template


def test_r01_cash_on_desk_when_cashier_leaves():
    c = evaluate([RULES["R01"]], ctx(people([Event(9.0, "cashier_left", 1)], away_from=9.0), {"F_cash_desk": [iv(8.6, 30)]}))
    assert len(c) == 1 and c[0].rule_id == "R01" and c[0].t == 9.0 and c[0].strong


def test_r01_silent_when_desk_empty():
    assert evaluate([RULES["R01"]], ctx(people([Event(9.0, "cashier_left", 1)], away_from=9.0), {"F_cash_desk": []})) == []


def test_r02_open_safe_while_cashier_absent():
    c = evaluate([RULES["R02"]], ctx(people([], away_from=10.0), {"F_safe_open": [iv(2.0, 20.0)]}))
    assert [x.t for x in c] == [10.0]


def test_r02_open_safe_with_cashier_present_is_fine():
    assert evaluate([RULES["R02"]], ctx(people([]), {"F_safe_open": [iv(2.0, 20.0)]})) == []


def test_r05_left_while_client_present():
    p = people([Event(9.0, "cashier_left", 1)], away_from=9.0, others=lambda t: 1)
    assert [x.rule_id for x in evaluate([RULES["R05"]], ctx(p))] == ["R05"]


def test_r06_entry_with_cash_from_previous_operation():
    p = people([Event(12.0, "entered", 2)], others=lambda t: 1 if t >= 12.0 else 0)
    c = evaluate([RULES["R06"]], ctx(p, {"F_cash_desk": [iv(5.0, 20.0)]}))
    assert [x.t for x in c] == [12.0]


def test_r06_not_when_previous_client_still_present():
    p = people([Event(12.0, "entered", 3)], others=lambda t: 1)
    assert evaluate([RULES["R06"]], ctx(p, {"F_cash_desk": [iv(5.0, 20.0)]})) == []


def test_r08_door_open_long():
    assert [x.t for x in evaluate([RULES["R08"]], ctx(people([]), {"F_door_open": [iv(6.0, 21.0)]}))] == [6.0]
    assert evaluate([RULES["R08"]], ctx(people([]), {"F_door_open": [iv(6.0, 9.0)]})) == []


def test_r09_second_visitor_enters():
    p = people([Event(17.0, "entered", 3)], others=lambda t: 2 if t >= 17.0 else 1)
    assert [x.t for x in evaluate([RULES["R09"]], ctx(p))] == [17.0]


def test_r10_presence_needs_min_duration():
    assert [x.t for x in evaluate([RULES["R10"]], ctx(people([]), {"F_items": [iv(15.0, 25.0)]}))] == [15.0]
    assert evaluate([RULES["R10"]], ctx(people([]), {"F_items": [iv(15.0, 15.6)]})) == []


def test_quality_rule_dark_video():
    q = [(t, 5.0 if t >= 10.0 else 120.0, 50.0) for t in T]
    assert [x.t for x in evaluate([RULES["R18"]], ctx(people([]), quality=q))] == [10.0]
```

- [ ] **Step 3: Run** `pytest tests/test_rules.py -q` — Expected: FAIL (module missing).

- [ ] **Step 4: Implement `app/pipeline/rules.py`**

```python
"""S6: rule catalog + evaluation of CV events x VLM states -> violation candidates."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from app.pipeline.people import PeopleTimeline
from app.pipeline.timeline import Interval, Series, active


@dataclass
class Rule:
    id: str
    type: str
    families: list[str]
    facts: list[str]
    params: dict
    norm: dict
    strength: str
    template: str
    verify: list[dict] = field(default_factory=list)
    enabled: bool = True


@dataclass
class Candidate:
    rule_id: str
    t: float
    evidence_t: float
    fact: str | None
    score: float
    strong: bool = False


@dataclass
class RuleContext:
    people: PeopleTimeline
    states: dict[str, list[Interval]]
    series: dict[str, Series]
    quality: list[tuple[float, float, float]]     # (t, brightness, sharpness)
    duration: float


def load_rules(path: Path) -> list[Rule]:
    rules = []
    for d in yaml.safe_load(Path(path).read_text(encoding="utf-8")):
        fam = d.pop("family")
        d["families"] = fam if isinstance(fam, list) else [fam]
        rules.append(Rule(**d))
    return [r for r in rules if r.enabled]


def _cand(rule: Rule, fact: str, iv: Interval, t: float) -> Candidate:
    return Candidate(rule.id, t, iv.peak_t, fact, iv.peak_p, iv.strong >= rule.params.get("strong_k", 3))


def on_cashier_left(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    out = []
    for ev in ctx.people.events:
        if ev.kind != "cashier_left":
            continue
        for f in rule.facts:
            iv = active(ctx.states.get(f, []), ev.t, rule.params.get("tol", 1.5))
            if iv:
                out.append(_cand(rule, f, iv, ev.t))
                break
    return out


def unattended_state(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    out, min_s = [], rule.params.get("min_s", 3.0)
    for f in rule.facts:
        for iv in ctx.states.get(f, []):
            ts = [t for t in ctx.people.times if iv.start <= t <= iv.end and ctx.people.away_at(t)]
            if ts and ts[-1] - ts[0] >= min_s:
                out.append(_cand(rule, f, iv, ts[0]))
    return out


def cashier_left_with_client(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    stay = rule.params.get("client_stays_s", 2.0)
    return [Candidate(rule.id, ev.t, ev.t + 1.0, None, 0.7, False)
            for ev in ctx.people.events
            if ev.kind == "cashier_left" and ctx.people.others_at(ev.t) > 0 and ctx.people.others_at(ev.t + stay) > 0]


def on_entry(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    out, before = [], rule.params.get("before_s", 1.5)
    for ev in ctx.people.events:
        if ev.kind != "entered" or ctx.people.others_at(ev.t - before) > 0:
            continue                                   # previous visitor still here -> same operation
        for f in rule.facts:
            iv = active(ctx.states.get(f, []), ev.t - before / 2, before / 2)
            if iv and iv.start <= ev.t - 0.5:
                out.append(_cand(rule, f, iv, ev.t))
                break
    return out


def entry_during_service(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    stay = rule.params.get("min_stay_s", 1.0)
    return [Candidate(rule.id, ev.t, ev.t + 1.0, None, 0.6, False)
            for ev in ctx.people.events
            if ev.kind == "entered" and ctx.people.others_at(ev.t - 1.0) >= 1 and ctx.people.others_at(ev.t + stay) >= 2]


def duration(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    min_s = rule.params.get("min_s", 8.0)
    return [_cand(rule, f, iv, iv.start) for f in rule.facts for iv in ctx.states.get(f, []) if iv.length >= min_s]


def presence(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    min_s = rule.params.get("min_s", 2.0)
    return [_cand(rule, f, iv, iv.start) for f in rule.facts for iv in ctx.states.get(f, []) if iv.length >= min_s]


def quality(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    p = rule.params
    bad = [b < p.get("dark", 25.0) or s < p.get("flat", 10.0) for _, b, s in ctx.quality]
    out, start = [], None
    for (t, _, _), is_bad in zip(ctx.quality + [(ctx.duration + 1, 255.0, 255.0)], bad + [False]):
        if is_bad and start is None:
            start = t
        elif not is_bad and start is not None:
            if t - start >= p.get("min_s", 2.0):
                out.append(Candidate(rule.id, start, start, None, 0.9, True))
            start = None
    return out


FAMILIES: dict[str, Callable[[Rule, RuleContext], list[Candidate]]] = {
    "on_cashier_left": on_cashier_left,
    "unattended_state": unattended_state,
    "cashier_left_with_client": cashier_left_with_client,
    "on_entry": on_entry,
    "entry_during_service": entry_during_service,
    "duration": duration,
    "presence": presence,
    "quality": quality,
}


def evaluate(rules: list[Rule], ctx: RuleContext) -> list[Candidate]:
    return [c for r in rules for fam in r.families for c in FAMILIES[fam](r, ctx)]
```

- [ ] **Step 5: Run** `pytest tests/test_rules.py -q` — Expected: `12 passed`.

- [ ] **Step 6: Checkpoint** — `pytest -q` green.

---

### Task 14: Candidate verification — S7 (Track C)

**Files:**
- Create: `app/pipeline/verify.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Consumes: `Candidate`, `Rule` (Task 13); `VideoFrames`, `crop` (Task 5); `Zones` (Task 10); `Fact` (Task 11); `VLM` (Task 9).
- Produces: `clip_times(t, window, duration, step=1.0, max_frames=8) -> list[float]`; `build_items(c, rule, frames, zones, facts_by_id) -> list[tuple[list[np.ndarray], str]]`; `decide(rule, probs) -> tuple[bool, float]`; `verify_all(cands, rules_by_id, frames, zones, facts_by_id, vlm) -> list[tuple[Candidate, bool, float]]`.

- [ ] **Step 1: Write failing tests `tests/test_verify.py`**

```python
import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.probes import Fact
from app.pipeline.rules import Candidate, Rule
from app.pipeline.scene import Zones
from app.pipeline.verify import clip_times, decide, verify_all
from app.pipeline.vlm import FakeVLM

RULE = Rule("R01", "t", ["on_cashier_left"], ["F_cash_desk"], {}, {"act": "a", "point": "p"}, "В", "{t}",
            verify=[{"q": "money A?", "window": [0, 5]}, {"q": "money B?", "window": [0, 5]},
                    {"q": "seat empty?", "window": [1, 6], "zone": "full", "expect": "yes"}])
FACTS = {"F_cash_desk": Fact("F_cash_desk", "desk", "q")}
VF = VideoFrames([i * 0.2 for i in range(100)], [np.zeros((50, 30, 3), np.uint8)] * 100, 20.0)


def test_clip_times_clamped_and_capped():
    assert clip_times(1.0, [-3, 5], 20.0, step=1.0, max_frames=8) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert len(clip_times(10.0, [-30, 30], 20.0, step=0.5, max_frames=8)) == 8


def test_decide_majority_with_expect_no():
    rule = Rule("X", "t", [], [], {}, {}, "В", "{t}", verify=[{"q": "a"}, {"q": "b"}, {"q": "c", "expect": "no"}])
    assert decide(rule, [0.9, 0.8, 0.1])[0] is True
    assert decide(rule, [0.9, 0.2, 0.9])[0] is False


def test_verify_all_confirms_and_rejects():
    c = Candidate("R01", 9.0, 10.0, "F_cash_desk", 0.9)
    yes = verify_all([c], {"R01": RULE}, VF, Zones(desk=(0, 0.5, 1, 1)), FACTS, FakeVLM(answer=lambda i, q: 0.9))
    no = verify_all([c], {"R01": RULE}, VF, Zones(desk=(0, 0.5, 1, 1)), FACTS, FakeVLM(answer=lambda i, q: 0.1))
    assert yes[0][1] is True and no[0][1] is False


def test_rule_without_verify_passes_through():
    r = Rule("R18", "t", ["quality"], [], {}, {}, "Н", "{t}", verify=[])
    c = Candidate("R18", 3.0, 3.0, None, 0.9, True)
    out = verify_all([c], {"R18": r}, VF, Zones(), FACTS, FakeVLM())
    assert out[0][1] is True and out[0][2] == 0.9
```

- [ ] **Step 2: Run** `pytest tests/test_verify.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `app/pipeline/verify.py`**

```python
"""S7: re-check each candidate on a short clip of zone crops with 2-3 phrasings, majority vote."""
import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import crop
from app.pipeline.probes import Fact
from app.pipeline.rules import Candidate, Rule
from app.pipeline.scene import Zones
from app.pipeline.vlm import VLM

CLIP_NOTE = " The images are consecutive video frames in time order."


def clip_times(t: float, window: list[float], duration: float, step: float = 1.0, max_frames: int = 8) -> list[float]:
    a, b = max(0.0, t + window[0]), min(duration, t + window[1])
    ts, x = [], a
    while x <= b + 1e-6:
        ts.append(round(x, 2))
        x += step
    if len(ts) > max_frames:
        ts = [ts[int(i * len(ts) / max_frames)] for i in range(max_frames)]
    return ts or [round(min(max(t, 0.0), duration), 2)]


def build_items(c: Candidate, rule: Rule, frames: VideoFrames, zones: Zones,
                facts_by_id: dict[str, Fact]) -> list[tuple[list[np.ndarray], str]]:
    default_zone = facts_by_id[c.fact].zone if c.fact in facts_by_id else "full"
    items = []
    for v in rule.verify:
        box = zones.get(v.get("zone", default_zone))
        imgs = [crop(frames.at(x), box) for x in clip_times(c.t, v.get("window", [-3, 5]), frames.duration)]
        items.append((imgs, v["q"] + CLIP_NOTE))
    return items


def decide(rule: Rule, probs: list[float]) -> tuple[bool, float]:
    agree = [p if v.get("expect", "yes") == "yes" else 1.0 - p for v, p in zip(rule.verify, probs)]
    votes = sum(a > 0.5 for a in agree)
    return votes * 2 > len(agree), sum(agree) / len(agree)


def verify_all(cands: list[Candidate], rules_by_id: dict[str, Rule], frames: VideoFrames, zones: Zones,
               facts_by_id: dict[str, Fact], vlm: VLM) -> list[tuple[Candidate, bool, float]]:
    items, owners = [], []
    for i, c in enumerate(cands):
        for it in build_items(c, rules_by_id[c.rule_id], frames, zones, facts_by_id):
            items.append(it)
            owners.append(i)
    probs = vlm.yes_probs(items) if items else []
    per: dict[int, list[float]] = {}
    for i, p in zip(owners, probs):
        per.setdefault(i, []).append(p)
    out = []
    for i, c in enumerate(cands):
        rule = rules_by_id[c.rule_id]
        if not rule.verify:
            out.append((c, True, c.score))
        else:
            ok, score = decide(rule, per[i])
            out.append((c, ok, score))
    return out
```

- [ ] **Step 4: Run** `pytest tests/test_verify.py -q` — Expected: `4 passed`.

- [ ] **Step 5: Checkpoint** — `pytest -q` green.

---

### Task 15: Report builder and pipeline orchestrator (Integration)

**Files:**
- Create: `app/pipeline/report.py`, `app/pipeline/run.py`
- Modify: `app/main.py` (use `build_processor` unless `PIPELINE=stub`)
- Test: `tests/test_report.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: everything from Tasks 5–14; `Violation`, `EvidenceFrame`, `RuleRef`, `JobResult` (Task 1).
- Produces: `jpeg_b64(img, quality=85) -> str`; `build_violations(accepted: list[tuple[Candidate, float]], rules_by_id, frames, merge_s=5.0) -> list[Violation]`; `Models(vlm, detector)`; `build_models(settings) -> Models`; `analyze(path, models, settings, rules, facts) -> tuple[str, list[Violation], dict]`; `build_processor(settings, models=None) -> ProcessFn`.

- [ ] **Step 1: Write failing tests `tests/test_report.py`**

```python
import base64

import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.report import build_violations, jpeg_b64
from app.pipeline.rules import Candidate, Rule

VF = VideoFrames([i * 0.2 for i in range(100)], [np.full((60, 40, 3), 100, np.uint8)] * 100, 20.0)
R01 = Rule("R01", "Наличные без присмотра", ["on_cashier_left"], ["F_cash_desk"], {"offset_s": 0.0},
           {"act": "Правила № 231", "point": "п. 20 ч. 2", "url": "u"}, "В", "В {t} с кассир ушёл (кадр {te} с).")
R08 = Rule("R08", "Дверь открыта", ["duration"], ["F_door_open"], {}, {"act": "Правила № 14", "point": "п. 16"}, "Н", "С {t} с дверь открыта.")


def test_jpeg_b64_is_jpeg():
    assert base64.b64decode(jpeg_b64(np.zeros((10, 10, 3), np.uint8)))[:2] == b"\xff\xd8"


def test_build_violation_fields_and_frames():
    v = build_violations([(Candidate("R01", 9.4, 10.2, "F_cash_desk", 0.9), 0.85)], {"R01": R01}, VF)
    assert len(v) == 1
    assert v[0].type == "Наличные без присмотра" and v[0].timecode_sec == 9.4
    assert "9.4" in v[0].description and "Правила № 231, п. 20 ч. 2" in v[0].description
    assert 1 <= len(v[0].evidence_frames) <= 3 and v[0].evidence_frames[0].timestamp_sec == 9.4
    assert v[0].rule.point == "п. 20 ч. 2"


def test_merge_same_rule_within_window_keeps_best():
    a = (Candidate("R01", 9.0, 9.0, "F_cash_desk", 0.7), 0.6)
    b = (Candidate("R01", 11.0, 11.0, "F_cash_desk", 0.9), 0.9)
    c = (Candidate("R01", 18.0, 18.0, "F_cash_desk", 0.9), 0.8)
    v = build_violations([a, b, c], {"R01": R01}, VF)
    assert [x.timecode_sec for x in v] == [11.0, 18.0]


def test_indirect_norm_is_phrased_as_possible():
    v = build_violations([(Candidate("R08", 6.0, 8.0, "F_door_open", 0.9), 0.9)], {"R08": R08}, VF)
    assert v[0].description.startswith("Возможное несоответствие")
```

- [ ] **Step 2: Run** `pytest tests/test_report.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `app/pipeline/report.py`**

```python
"""S9: merge accepted candidates, pick evidence frames, write descriptions from templates."""
import base64
import io

import numpy as np
from PIL import Image

from app.pipeline.frames import VideoFrames
from app.pipeline.rules import Candidate, Rule
from app.schemas import EvidenceFrame, RuleRef, Violation


def jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _evidence_times(t: float, te: float, duration: float) -> list[float]:
    picked: list[float] = []
    for x in (t, te, min(duration, t + 3.0)):
        x = round(min(max(x, 0.0), duration), 1)
        if all(abs(x - p) >= 0.5 for p in picked):
            picked.append(x)
    return picked[:3]


def build_violations(accepted: list[tuple[Candidate, float]], rules_by_id: dict[str, Rule],
                     frames: VideoFrames, merge_s: float = 5.0) -> list[Violation]:
    best: list[tuple[Candidate, float]] = []
    for c, score in sorted(accepted, key=lambda x: x[0].t):
        prev = next((i for i, (b, _) in enumerate(best) if b.rule_id == c.rule_id and abs(b.t - c.t) <= merge_s), None)
        if prev is None:
            best.append((c, score))
        elif score > best[prev][1]:
            best[prev] = (c, score)
    out = []
    for c, score in sorted(best, key=lambda x: -x[1]):
        rule = rules_by_id[c.rule_id]
        t = round(max(0.0, c.t + rule.params.get("offset_s", 0.0)), 1)
        te = round(c.evidence_t, 1)
        text = rule.template.format(t=f"{t:.1f}", te=f"{te:.1f}")
        if rule.strength == "Н":
            text = "Возможное несоответствие: " + text
        text += f" Норматив: {rule.norm['act']}, {rule.norm['point']}."
        frames_b64 = [EvidenceFrame(timestamp_sec=x, image_base64=jpeg_b64(frames.at(x))) for x in _evidence_times(t, te, frames.duration)]
        out.append(Violation(type=rule.type, timecode_sec=t, description=text, evidence_frames=frames_b64,
                             rule_id=rule.id, confidence=round(score, 3),
                             rule=RuleRef(act=rule.norm["act"], point=rule.norm["point"], strength=rule.strength, url=rule.norm.get("url"))))
    return out
```

- [ ] **Step 4: Run** `pytest tests/test_report.py -q` — Expected: `4 passed`.

- [ ] **Step 5: Write failing tests `tests/test_run.py`**

```python
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.pipeline.probes import load_facts
from app.pipeline.rules import load_rules
from app.pipeline.run import Models, analyze, build_processor
from app.pipeline.vlm import FakeVLM

W, H = 96, 160


def painter(i):
    """Grey room (not dark, so the camera-covered rule stays silent). Frames 0-269 (9 s @30fps): cashier
    (red block) seated; afterwards gone. Cash (white block) on the desk from 6 s."""
    img = np.full((H, W, 3), 120, np.uint8)
    if i < 270:
        img[40:100, 30:60] = (200, 0, 0)
    if i >= 180:
        img[110:130, 10:40] = 255
    return img


class BlockDetector:
    def __call__(self, images):
        return [np.array([[30, 40, 60, 100, 0.95]], np.float32) if im[70, 45, 0] > 150 and im[70, 45, 1] < 80
                else np.zeros((0, 5), np.float32) for im in images]


def has_white(img):
    return bool((img.min(axis=2) > 200).any())


def cash_vlm():
    def answer(images, q):
        if "banknotes" in q or "money" in q.lower() or "cash" in q.lower():
            return 0.95 if has_white(images[0]) else 0.05
        return 0.05
    return FakeVLM(answer=answer, reply=lambda imgs, p: '{"bbox_2d": [0, 600, 1000, 900]}' if "desk" in p else '{"bbox_2d": null}')


SETTINGS = replace(Settings(), det_fps=5.0, time_budget_s=120.0)
RULES = load_rules(Path("app/rules.yaml"))
FACTS = load_facts(Path("app/facts.yaml"))


def test_cash_left_on_desk_is_reported(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))                 # 15 s
    verdict, violations, debug = analyze(path, Models(cash_vlm(), BlockDetector()), SETTINGS, RULES, FACTS)
    assert verdict == "violation"
    r01 = [v for v in violations if v.rule_id == "R01"]
    assert r01 and abs(r01[0].timecode_sec - 9.0) <= 1.0
    assert 1 <= len(r01[0].evidence_frames) <= 3


def test_clean_video_no_violation(tmp_video):
    path = tmp_video(150, 30, lambda i: painter(0), size=(W, H))     # cashier seated, no cash
    verdict, violations, _ = analyze(path, Models(cash_vlm(), BlockDetector()), SETTINGS, RULES, FACTS)
    assert verdict == "no_violation" and violations == []


def test_budget_skips_verification(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))
    s = replace(SETTINGS, time_budget_s=0.0)
    verdict, violations, debug = analyze(path, Models(cash_vlm(), BlockDetector()), s, RULES, FACTS)
    assert debug["verification_skipped"] is True
    assert verdict == "violation"                                    # strong probe candidates still accepted


def test_process_rejects_non_video(tmp_path):
    bad = tmp_path / "x.mov"
    bad.write_bytes(b"not a video")
    process = build_processor(SETTINGS, Models(cash_vlm(), BlockDetector()))
    with pytest.raises(Exception):
        process(bad, "job1")
```

- [ ] **Step 6: Run** `pytest tests/test_run.py -q` — Expected: FAIL (module missing).

- [ ] **Step 7: Implement `app/pipeline/run.py`**

```python
"""Orchestrator: S0..S9 with a time budget."""
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.jobs import ProcessFn
from app.pipeline.change import change_times, dense_times, zone_change
from app.pipeline.frames import load_frames
from app.pipeline.people import TorchvisionPersonDetector, analyze_people, track_people
from app.pipeline.probes import Fact, load_facts, probe_times, run_probes
from app.pipeline.report import build_violations
from app.pipeline.rules import Rule, RuleContext, evaluate, load_rules
from app.pipeline.scene import locate_zones
from app.pipeline.timeline import intervals
from app.pipeline.verify import verify_all
from app.pipeline.vlm import VLM, FakeVLM, OpenAIVLM
from app.schemas import JobResult, Violation

APP_DIR = Path(__file__).resolve().parents[1]
SCENE_THR, HAND_THR = 0.03, 0.05          # probe 2026-10-17; recalibrate in Task 16


@dataclass
class Models:
    vlm: VLM
    detector: object


def build_models(settings: Settings) -> Models:
    vlm = FakeVLM() if settings.vlm_backend == "fake" else OpenAIVLM(settings.vlm_url, settings.vlm_model, settings.vlm_workers)
    return Models(vlm, TorchvisionPersonDetector(settings.detector, settings.device))


def analyze(path: Path, models: Models, settings: Settings, rules: list[Rule], facts: list[Fact]) -> tuple[str, list[Violation], dict]:
    t0 = time.monotonic()
    debug: dict = {"verification_skipped": False}
    frames = load_frames(path, settings.det_fps)
    if not frames.images:
        raise ValueError("no video frames could be decoded")
    h, w = frames.images[0].shape[:2]
    zones = locate_zones(frames.at(min(1.0, frames.duration / 2)), models.vlm, settings.vlm_coord_mode)
    tracks = track_people(frames, models.detector, settings.det_fps)
    people = analyze_people(tracks, frames.times, zones.desk, zones.safe, frames.duration, w / h)

    extra = [e.t for e in people.events]
    for box in (zones.desk, zones.safe, zones.door):
        extra += dense_times(change_times(zone_change(frames, box, tracks, mask_people=True), SCENE_THR))
    extra += dense_times(change_times(zone_change(frames, zones.desk, tracks, mask_people=False), HAND_THR))
    times = probe_times(frames.duration, settings.probe_fps, extra, settings.max_probe_frames)
    series = run_probes(frames, zones, models.vlm, facts, times)
    states = {fid: intervals(s) for fid, s in series.items()}
    debug.update(zones=zones.__dict__, events=[e.__dict__ for e in people.events], probe_frames=len(times),
                 t_probes=round(time.monotonic() - t0, 1))

    quality = list(zip(frames.times, frames.brightness, frames.sharpness))
    cands = evaluate(rules, RuleContext(people, states, series, quality, frames.duration))
    rules_by_id = {r.id: r for r in rules}
    if time.monotonic() - t0 > settings.time_budget_s:
        debug["verification_skipped"] = True
        checked = [(c, c.strong, c.score) for c in cands]
    else:
        checked = verify_all(cands, rules_by_id, frames, zones, {f.id: f for f in facts}, models.vlm)
    accepted = [(c, score) for c, ok, score in checked if ok or c.strong]
    violations = build_violations(accepted, rules_by_id, frames)
    debug.update(candidates=len(cands), accepted=len(accepted), t_total=round(time.monotonic() - t0, 1))
    return ("violation" if violations else "no_violation"), violations, debug


def build_processor(settings: Settings, models: Models | None = None) -> ProcessFn:
    models = models or build_models(settings)
    rules = load_rules(APP_DIR / "rules.yaml")
    facts = load_facts(APP_DIR / "facts.yaml")

    def process(path: Path, job_id: str) -> JobResult:
        verdict, violations, _ = analyze(path, models, settings, rules, facts)
        return JobResult(job_id=job_id, status="done", verdict=verdict, violations=violations)

    return process
```

- [ ] **Step 8: Run** `pytest tests/test_run.py -q` — Expected: `4 passed`. If `test_cash_left_on_desk_is_reported` reports no `cashier_left`, print `debug["events"]`: the block detector must return a box only while the red block is present.

- [ ] **Step 9: Wire into `app/main.py`**

```python
import os

from app.api import create_app
from app.config import load_settings
from app.stub import stub_process

settings = load_settings()
if os.environ.get("PIPELINE", "real") == "stub":
    process = stub_process
else:
    from app.pipeline.run import build_processor
    process = build_processor(settings)
app = create_app(settings, process)
```

- [ ] **Step 10: Checkpoint** — `pytest -q` green. Deploy to host: copy code, restart `deploy/start.sh` (without `PIPELINE=stub`), process `X-26.mov` via public URL → `violation` with R01 near 9–10 s.

---

### Task 16: Evaluation harness and calibration (Integration, GPU host)

**Files:**
- Create: `scripts/labels.py`, `scripts/eval.py`, `data/types.csv`, `data/layouts.csv`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `analyze`, `build_models`, `load_rules`, `load_facts`, `load_settings`.
- Produces: `load_labels(xlsx) -> dict[str, list[float]]` (keys `X-1…`, Latin X, duplicates merged); `score(preds, labels, types, layouts) -> dict`; CLI `python -m scripts.eval --videos open/videos --labels open/open_labels.xlsx --out eval_out [--only X-26,X-25]`.

- [ ] **Step 1: Create `data/types.csv`** (our hypotheses from `docs/dataset_analysis.md`; `-` = deferred type)

```
video,t,rule_id
X-1,22,-
X-2,10,R14
X-4,57,R06
X-4,68,R07
X-11,18,R04
X-13,47,R16
X-15,15,R01
X-17,7,R15
X-18,20,R09
X-20,5,R02
X-22,14,R03
X-24,8,R17
X-26,9,R01
X-28,7,R04
X-30,13,R09
X-32,5,R08
X-33,6,R08
X-34,13,R13
X-38,33,R06
X-40,17,R10
X-44,16,-
X-46,11,R12
```

- [ ] **Step 2: Create `data/layouts.csv`** (from the first-frame grid, 2026-10-17)

```
video,layout
X-1,A
X-2,A
X-3,A
X-4,A
X-5,A
X-16,A
X-17,A
X-18,A
X-19,A
X-8,B
X-10,B
X-11,B
X-12,B
X-13,B
X-14,B
X-15,B
X-20,B
X-21,B
X-22,B
X-23,B
X-24,B
X-25,B
X-26,B
X-27,B
X-28,B
X-29,B
X-30,C
X-31,C
X-32,C
X-34,D
X-35,D
X-37,D
X-38,D
X-39,D
X-33,E
X-40,E
X-43,E
X-44,E
X-45,E
X-46,E
```

- [ ] **Step 3: Write failing tests `tests/test_eval.py`**

```python
from pathlib import Path

from app.schemas import Violation
from scripts.eval import score
from scripts.labels import load_labels


def V(t, rid):
    return Violation(type=rid, timecode_sec=t, description="", evidence_frames=[], rule_id=rid)


def test_labels_parse_and_dedupe():
    labels = load_labels(Path("open/open_labels.xlsx"))
    assert labels["X-1"] == [22.0] and labels["X-3"] == []
    assert sorted(labels["X-4"]) == [57.0, 68.0] and "X-47" not in labels and "X-48" not in labels
    assert len(labels) == 40


def test_score_counts_recall_fp_time_and_type():
    labels = {"A": [10.0], "B": [], "C": [5.0]}
    preds = {"A": [V(10.8, "R01")], "B": [V(3.0, "R08")], "C": []}
    types = {("A", 10.0): "R01", ("C", 5.0): "R08"}
    layouts = {"A": "L1", "B": "L1", "C": "L2"}
    s = score(preds, labels, types, layouts)
    assert s["all"]["violation_videos"] == 2 and s["all"]["caught"] == 1
    assert s["all"]["clean_videos"] == 1 and s["all"]["false_alarms"] == 1
    assert s["all"]["time_within_2s"] == 1 and s["all"]["type_correct"] == 1
    assert s["L2"]["caught"] == 0
```

- [ ] **Step 4: Run** `pytest tests/test_eval.py -q` — Expected: FAIL (modules missing).

- [ ] **Step 5: Implement `scripts/labels.py`**

```python
"""Parse open/open_labels.xlsx without openpyxl. Keys use Latin 'X-'; duplicates X-47→X-4, X-48→X-5 merged."""
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}


def parse_time(text: str) -> float | None:
    m = re.match(r"\s*(?:(\d+)\s*минут\w*\s*)?(\d+)\s*секунд", text or "")
    return None if not m else float(int(m.group(1) or 0) * 60 + int(m.group(2)))


def load_labels(path: Path) -> dict[str, list[float]]:
    z = zipfile.ZipFile(path)
    labels: dict[str, list[float]] = {}
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        cells = {c.get("r")[0]: "".join(x.text or "" for x in c.iter(NS + "t")) for c in row.findall(NS + "c")}
        name = cells.get("A", "").replace("Х", "X").strip()
        if not name.startswith("X-"):
            continue
        key = DUPLICATES.get(name, name)
        labels.setdefault(key, [])
        t = parse_time(cells.get("B", ""))
        if t is not None and t not in labels[key]:
            labels[key].append(t)
    return labels


def video_key(filename: str) -> str:
    return Path(filename).stem.replace("Х", "X")
```

- [ ] **Step 6: Implement `scripts/eval.py`**

```python
"""Run the pipeline on the training videos and report jury-like metrics per camera layout."""
import argparse
import csv
import json
import time
from pathlib import Path

from app.config import load_settings
from app.pipeline.probes import load_facts
from app.pipeline.rules import load_rules
from app.pipeline.run import APP_DIR, analyze, build_models
from app.schemas import Violation
from scripts.labels import DUPLICATES, load_labels, video_key


def score(preds: dict[str, list[Violation]], labels: dict[str, list[float]],
          types: dict[tuple[str, float], str], layouts: dict[str, str]) -> dict:
    groups: dict[str, dict] = {}

    def bump(g, k, n=1):
        groups.setdefault(g, {"violation_videos": 0, "caught": 0, "clean_videos": 0, "false_alarms": 0,
                              "labels": 0, "time_within_2s": 0, "type_correct": 0, "abs_dt": []})
        if k == "abs_dt":
            groups[g][k].append(n)
        else:
            groups[g][k] += n

    for name, ts in labels.items():
        if name not in preds:
            continue
        p = preds[name]
        for g in ("all", layouts.get(name, "?")):
            if ts:
                bump(g, "violation_videos")
                bump(g, "caught", int(bool(p)))
                for t in ts:
                    bump(g, "labels")
                    if p:
                        near = min(p, key=lambda v: abs(v.timecode_sec - t))
                        bump(g, "abs_dt", abs(near.timecode_sec - t))
                        bump(g, "time_within_2s", int(abs(near.timecode_sec - t) <= 2.0))
                        bump(g, "type_correct", int(near.rule_id == types.get((name, t))))
            else:
                bump(g, "clean_videos")
                bump(g, "false_alarms", int(bool(p)))
    for g in groups.values():
        dts = sorted(g.pop("abs_dt"))
        g["median_abs_dt"] = round(dts[len(dts) // 2], 2) if dts else None
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="open/videos")
    ap.add_argument("--labels", default="open/open_labels.xlsx")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    settings = load_settings()
    models = build_models(settings)
    rules, facts = load_rules(APP_DIR / "rules.yaml"), load_facts(APP_DIR / "facts.yaml")
    labels = load_labels(Path(args.labels))
    types = {(r["video"], float(r["t"])): r["rule_id"] for r in csv.DictReader(open("data/types.csv"))}
    layouts = {r["video"]: r["layout"] for r in csv.DictReader(open("data/layouts.csv"))}
    only = set(filter(None, args.only.split(",")))
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    preds: dict[str, list[Violation]] = {}
    for f in sorted(Path(args.videos).glob("*.mov")):
        name = video_key(f.name)
        if name in DUPLICATES or (only and name not in only):
            continue
        t0 = time.monotonic()
        verdict, violations, debug = analyze(f, models, settings, rules, facts)
        preds[name] = violations
        (out / f"{name}.json").write_text(json.dumps({"verdict": verdict, "debug": debug, "violations": [
            {k: v for k, v in x.model_dump().items() if k != "evidence_frames"} for x in violations]}, ensure_ascii=False, indent=1, default=str))
        print(f"{name:6} {layouts.get(name, '?')} {verdict:12} {round(time.monotonic() - t0):>4}s labels={labels.get(name)} "
              f"pred={[(v.rule_id, v.timecode_sec) for v in violations]}", flush=True)
    print(json.dumps(score(preds, labels, types, layouts), indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run** `pytest tests/test_eval.py -q` — Expected: `2 passed`.

- [ ] **Step 8: First full run on the GPU host**

Run: `DETECTOR=frcnn_r50v2 DEVICE=cuda VLM_COORD_MODE=<from Task 10> .venv/bin/python -m scripts.eval --out eval_out/run1 | tee eval_out/run1.log`
Expected: one line per video, then metrics JSON. Record: caught / 21, false alarms / 19, median |Δt|, max seconds per video.

- [ ] **Step 9: Calibration loop (repeat until targets in spec §1 or time runs out)**

For each miss or false alarm, open `eval_out/run1/<video>.json` → `debug.events`, `debug.zones`, candidates, and classify the failure layer: zones (S1) / events (S2) / probes (S4 series) / rule params / verification. Adjust only:
- `app/rules.yaml` `params` (`min_s`, `tol`, `before_s`, `strong_k`, `offset_s`);
- `app/pipeline/timeline.py` `on/off` defaults;
- `app/pipeline/people.py` `AWAY_D`, `AWAY_S`, `CHAIN_GAP_S`;
- fact questions in `app/facts.yaml` (wording only).

Tune using layouts A, B, D, E; then read layout C metrics unchanged as the held-out check (repeat rotating the held-out layout once if time allows). Set per-rule `offset_s` = median(label − predicted time) for that rule over the training layouts.

- [ ] **Step 10: Speed check** — the slowest video (X-14, 142 s) must finish < 120 s; submit 5 videos at once via the API and confirm all finish < 5 min. If too slow: `max_probe_frames: 160`, `probe_fps: 0.5` plus change-driven extras, or `VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct`.

---

### Task 17 (stretch): Tier-2 procedural rules

**Files:**
- Modify: `app/facts.yaml`, `app/rules.yaml`, `app/pipeline/rules.py`
- Test: `tests/test_rules_tier2.py`

**Interfaces:**
- Consumes: `RuleContext`, `Candidate`, `active`, `FAMILIES` (Task 13).
- Produces: families `count_after_client_left` (R13), `handover_without_id` (R14), `collector_quick_leave` (R15).

- [ ] **Step 1: Add facts to `app/facts.yaml`**

```yaml
- id: F_counting
  zone: desk
  question: "Is the cashier counting banknotes by hand or feeding them into the banknote counting machine?"
- id: F_id_card
  zone: desk
  question: "Is an ID card or identity document being shown or lying on the desk?"
```

- [ ] **Step 2: Add rules to `app/rules.yaml`**

```yaml
- id: R13
  type: "Пересчёт наличных не в присутствии клиента"
  family: count_after_client_left
  facts: [F_counting]
  params: {after_s: 1.0, strong_k: 3}
  strength: "В"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 20 ч. 1; п. 31 пп. 4)", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента"}
  template: "В {t} с клиент покинул помещение, после чего кассир пересчитывает наличные (кадр {te} с)."
  verify:
    - {q: "Is the cashier counting money?", window: [0, 4]}
    - {q: "Is the cashier using the banknote counting machine?", window: [0, 4]}
    - {q: "Is the client absent while money is being counted?", window: [0, 4], zone: full}
- id: R14
  type: "Документ клиента на видео не зафиксирован"
  family: handover_without_id
  facts: [F_handover]
  params: {lookback_s: 20.0, strong_k: 3}
  strength: "С"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 19 пп. 1)", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "проводит визуальную идентификацию клиента, вносящего наличные деньги, с документом, удостоверяющим его личность"}
  template: "В {t} с клиент передаёт наличные; предъявление документа, удостоверяющего личность, на видео не зафиксировано."
  verify:
    - {q: "Is a client handing money to the cashier?", window: [-1, 2]}
    - {q: "Is there no ID card visible on the desk?", window: [-1, 2]}
    - {q: "Is cash being passed across the desk?", window: [-1, 2]}
- id: R15
  type: "Сумка инкассации принята без видимой проверки"
  family: collector_quick_leave
  facts: [F_bag_desk]
  params: {max_stay_s: 10.0, strong_k: 2}
  strength: "С"
  norm: {act: "Правила № 231 НБРК от 29.11.2019", point: "п. 24", url: "https://adilet.zan.kz/rus/docs/V1900019680",
         quote: "проверяет целостность сумки; проверяет правильность пломбирования сумки"}
  template: "В {t} с инкассатор оставляет сумку и уходит в течение нескольких секунд; проверка целостности сумки и пломбы при нём на видео не зафиксирована."
  verify:
    - {q: "Is a cash collection bag lying on the desk?", window: [0, 3]}
    - {q: "Did a person drop a bag on the desk and leave?", window: [-2, 3], zone: full}
    - {q: "Is a sealed bag on the table?", window: [0, 3]}
```

- [ ] **Step 3: Write failing tests `tests/test_rules_tier2.py`**

```python
from pathlib import Path

from app.pipeline.people import Event, PeopleTimeline
from app.pipeline.rules import RuleContext, evaluate, load_rules
from app.pipeline.timeline import Interval

T = [round(i * 0.2, 3) for i in range(150)]
RULES = {r.id: r for r in load_rules(Path("app/rules.yaml"))}


def ctx(events, states, others=lambda t: 0):
    p = PeopleTimeline(T, 1, [False] * len(T), [others(t) for t in T], events)
    return RuleContext(p, states, {}, [(t, 120.0, 50.0) for t in T], 30.0)


def iv(a, b):
    return Interval(a, b, a, 0.9, 5)


def test_r13_counting_after_client_left():
    c = evaluate([RULES["R13"]], ctx([Event(11.0, "exited", 2)], {"F_counting": [iv(12.5, 16.0)]}))
    assert [x.t for x in c] == [12.5]


def test_r13_not_when_counting_before_exit():
    assert evaluate([RULES["R13"]], ctx([Event(11.0, "exited", 2)], {"F_counting": [iv(5.0, 9.0)]})) == []


def test_r14_handover_without_id():
    c = evaluate([RULES["R14"]], ctx([], {"F_handover": [iv(10.0, 11.0)], "F_id_card": []}))
    assert [x.t for x in c] == [10.0]
    assert evaluate([RULES["R14"]], ctx([], {"F_handover": [iv(10.0, 11.0)], "F_id_card": [iv(4.0, 8.0)]})) == []


def test_r15_collector_leaves_quickly():
    events = [Event(3.0, "entered", 5), Event(9.0, "exited", 5)]
    c = evaluate([RULES["R15"]], ctx(events, {"F_bag_desk": [iv(6.5, 30.0)]}))
    assert [x.t for x in c] == [6.5]
```

- [ ] **Step 4: Run** `pytest tests/test_rules_tier2.py -q` — Expected: FAIL (`KeyError: 'count_after_client_left'`).

- [ ] **Step 5: Add families to `app/pipeline/rules.py`** (above `FAMILIES`, then register them)

```python
def count_after_client_left(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    out = []
    for ev in ctx.people.events:
        if ev.kind != "exited":
            continue
        for f in rule.facts:
            for iv in ctx.states.get(f, []):
                if ev.t + rule.params.get("after_s", 1.0) <= iv.start <= ev.t + 10.0:
                    out.append(_cand(rule, f, iv, iv.start))
    return out


def handover_without_id(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    look = rule.params.get("lookback_s", 20.0)
    ids = ctx.states.get("F_id_card", [])
    out = []
    for f in rule.facts:
        for iv in ctx.states.get(f, []):
            if not any(iv.start - look <= x.end and x.start <= iv.start for x in ids):
                out.append(_cand(rule, f, iv, iv.start))
    return out


def collector_quick_leave(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    max_stay = rule.params.get("max_stay_s", 10.0)
    entries = {e.track_id: e.t for e in ctx.people.events if e.kind == "entered"}
    out = []
    for ev in ctx.people.events:
        if ev.kind != "exited" or ev.track_id not in entries or ev.t - entries[ev.track_id] > max_stay:
            continue
        for f in rule.facts:
            for iv in ctx.states.get(f, []):
                if entries[ev.track_id] <= iv.start <= ev.t and iv.end >= ev.t:
                    out.append(_cand(rule, f, iv, iv.start))
    return out
```

and extend the registry:

```python
FAMILIES.update({
    "count_after_client_left": count_after_client_left,
    "handover_without_id": handover_without_id,
    "collector_quick_leave": collector_quick_leave,
})
```

- [ ] **Step 6: Run** `pytest tests/test_rules_tier2.py tests/test_rules.py -q` — Expected: all pass. Re-run `scripts.eval`; keep a tier-2 rule only if it does not add false alarms on clean videos (set `enabled: false` otherwise).

---

### Task 18 (stretch): Unknown-type channel — S8

**Files:**
- Create: `app/pipeline/unknown.py`
- Modify: `app/pipeline/run.py` (call after verification, only if time left)
- Test: `tests/test_unknown.py`

**Interfaces:**
- Consumes: `VideoFrames`, `PeopleTimeline`, `Rule`, `Candidate`, `VLM`, `clip_times`.
- Produces: `scan_unknown(frames, people, rules, vlm, already: set[str], threshold=0.85, max_events=6) -> list[tuple[Candidate, float]]`.

- [ ] **Step 1: Write failing test `tests/test_unknown.py`**

```python
import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.people import Event, PeopleTimeline
from app.pipeline.rules import Rule
from app.pipeline.unknown import scan_unknown
from app.pipeline.vlm import FakeVLM

VF = VideoFrames([i * 0.2 for i in range(100)], [np.zeros((40, 30, 3), np.uint8)] * 100, 20.0)
PEOPLE = PeopleTimeline(VF.times, 1, [False] * 100, [0] * 100, [Event(5.0, "entered", 2)])
R = Rule("R05", "Кассир покинул место", ["cashier_left_with_client"], [], {}, {"act": "a", "point": "p"}, "В", "{t}")


def test_description_matched_to_rule():
    vlm = FakeVLM(answer=lambda imgs, q: 0.95 if "R05" in q else 0.1, reply=lambda imgs, p: "The cashier walks away while a client waits.")
    out = scan_unknown(VF, PEOPLE, [R], vlm, already=set())
    assert [(c.rule_id, c.t) for c, _ in out] == [("R05", 5.0)]


def test_already_found_rules_skipped():
    vlm = FakeVLM(answer=lambda imgs, q: 0.95, reply=lambda imgs, p: "x")
    assert scan_unknown(VF, PEOPLE, [R], vlm, already={"R05"}) == []
```

- [ ] **Step 2: Run** `pytest tests/test_unknown.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `app/pipeline/unknown.py`**

```python
"""S8: describe event clips, then check the description against each catalog rule (describe -> match)."""
from app.pipeline.frames import VideoFrames
from app.pipeline.people import PeopleTimeline
from app.pipeline.rules import Candidate, Rule
from app.pipeline.verify import clip_times
from app.pipeline.vlm import VLM

DESCRIBE = ("These are consecutive frames from a bank cashier room camera. Describe in 2-3 sentences what the people do "
            "with money, bags, documents, the safe, the door and the laptop. Only state what is visible.")
MATCH = ("Description of a bank cashier room video: \"{desc}\"\n"
         "Rule {rid}: {rtype}.\nDoes the description show a breach of this rule?")


def scan_unknown(frames: VideoFrames, people: PeopleTimeline, rules: list[Rule], vlm: VLM,
                 already: set[str], threshold: float = 0.85, max_events: int = 6) -> list[tuple[Candidate, float]]:
    todo = [r for r in rules if r.id not in already]
    if not todo:
        return []
    out = []
    for ev in people.events[:max_events]:
        imgs = [frames.at(x) for x in clip_times(ev.t, [-2, 4], frames.duration, step=1.0, max_frames=6)]
        desc = vlm.text(imgs, DESCRIBE, 160).replace('"', "'")
        probs = vlm.yes_probs([([], MATCH.format(desc=desc, rid=r.id, rtype=r.type)) for r in todo])
        for r, p in zip(todo, probs):
            if p >= threshold:
                out.append((Candidate(r.id, ev.t, ev.t, None, p, False), p))
    return out
```

- [ ] **Step 4: Call it from `analyze` in `app/pipeline/run.py`** — after `accepted = ...`, insert:

```python
    if time.monotonic() - t0 < settings.time_budget_s * 0.7:
        from app.pipeline.unknown import scan_unknown
        accepted += scan_unknown(frames, people, rules, models.vlm, {c.rule_id for c, _ in accepted})
```

- [ ] **Step 5: Run** `pytest -q` — Expected: all pass. Re-run `scripts.eval`; disable the channel (`max_events=0`) if it adds false alarms on clean videos.

---

### Task 19: Submission documentation (morning 18.10)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`** with these sections, filled from the final eval run and catalog:
  1. **Адрес сервиса и API, ключ** — public URL, `X-API-Key` (sent to the jury separately), curl example from the TZ.
  2. **Инструкция запуска** — `deploy/setup.sh`, `deploy/start.sh`, env vars (`API_KEY`, `VLM_MODEL`, `VLM_COORD_MODE`, `DETECTOR`, `DEVICE`), offline mode (`HF_HUB_OFFLINE=1`).
  3. **Архитектура** — the S0–S9 diagram from `docs/methodology.md` §3 and the principle "время — из CV, смысл — из VLM, пункт — из каталога".
  4. **Модели и лицензии** — table from spec §12 (torchvision Faster R-CNN BSD-3, ByteTrack/`trackers` Apache-2.0, `supervision` MIT, Qwen3-VL Apache-2.0, vLLM Apache-2.0).
  5. **Нормативные источники и сопоставление** — acts A/B/G with adilet links; table rule id → type → clause → strength (generated from `app/rules.yaml`: `python -c "import yaml;[print(r['id'],r['type'],r['norm']['point'],r['strength'],sep=' | ') for r in yaml.safe_load(open('app/rules.yaml'))]"`).
  6. **Типы нарушений** — one line per enabled rule with what is visible.
  7. **Ограничения и известные ошибки** — from the last eval: missed labels and why (e.g. X-1, X-44 hand-level details at 464×848), false alarms, reflections, unseen camera angles, phone camera cannot be proven (п. 12), indirect norms (Н).
  8. **Качество на обучающих видео** — metrics JSON summary per layout.

- [ ] **Step 2: Final checks** — from a phone/another network: TZ curl → `job_id`, poll → `done` within 2 min; UI loads; `X-API-Key` missing → 401. Host balance ≥ 24 h of runtime. Do not restart the quick tunnel after submitting the URL.
