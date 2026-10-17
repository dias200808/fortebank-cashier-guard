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
    det_batch: int = 8
    probe_fps: float = 1.0
    max_probe_frames: int = 240
    time_budget_s: float = 200.0         # soft budget for one video's analysis
    job_limit_s: float = 270.0           # hard limit counted from submission (jury allows 300 s)
    unknown_channel: bool = True         # S8 describe -> match
    max_upload_mb: int = 300


def _flag(value: str) -> bool:
    return value.strip().lower() not in {"0", "false"}


def load_settings() -> Settings:
    e = os.environ
    return Settings(
        api_key=e.get("API_KEY", "dev-key"),
        jobs_dir=Path(e.get("JOBS_DIR", "jobs")),
        vlm_backend=e.get("VLM_BACKEND", "openai"),
        vlm_url=e.get("VLM_URL", "http://127.0.0.1:8000/v1"),
        vlm_model=e.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        vlm_coord_mode=e.get("VLM_COORD_MODE", "rel1000"),
        vlm_workers=int(e.get("VLM_WORKERS", "16")),
        detector=e.get("DETECTOR", "frcnn_mobile"),
        device=e.get("DEVICE", "cpu"),
        det_fps=float(e.get("DET_FPS", "5")),
        det_batch=int(e.get("DET_BATCH", "8")),
        probe_fps=float(e.get("PROBE_FPS", "1")),
        max_probe_frames=int(e.get("MAX_PROBE_FRAMES", "240")),
        time_budget_s=float(e.get("TIME_BUDGET_S", "200")),
        job_limit_s=float(e.get("JOB_LIMIT_S", "270")),
        unknown_channel=_flag(e.get("UNKNOWN_CHANNEL", "1")),
        max_upload_mb=int(e.get("MAX_UPLOAD_MB", "300")),
    )
