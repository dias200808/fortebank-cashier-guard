"""PIPELINE=demo: a UI showcase, not an analysis.

Every upload gets the same two example violations (rules R01 and R08 at ~25% and ~60% of the video)
with evidence frames cut from the uploaded video itself and a fixed example region (desk, door), so the
player, timecodes, region boxes and evidence cards can be shown without the GPU pipeline. Every type and description is labelled as a demo.
Imports stay light on purpose: no detector, tracker or VLM is loaded in this mode.
"""
import base64
import io
import time
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import yaml
from PIL import Image

from app.pipeline.frames import VideoFrames, load_frames
from app.schemas import EvidenceFrame, JobResult, RuleRef, Violation

RULES_PATH = Path(__file__).parent / "rules.yaml"
TYPE_PREFIX = "ДЕМО — "
DESC_PREFIX = "[Демо-режим: это не результат анализа] "
DEMO = (  # (rule id, position as a fraction of the duration, normalized [x1, y1, x2, y2] region)
    ("R01", 0.25, [0.06, 0.42, 0.72, 0.62]),   # where the cash desk sits in the open/ camera layouts
    ("R08", 0.60, [0.02, 0.02, 0.45, 0.45]),   # the glass entrance door, upper left
)
MAX_FRAMES = 60  # cap on decoded frames, so long uploads stay cheap in memory


def jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    """Same encoding as app.pipeline.report.jpeg_b64, without importing the pipeline's tracker stack."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=1)
def _rules() -> dict[str, dict]:
    return {r["id"]: r for r in yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))}


def _probe_duration(path: Path) -> float:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        if stream.duration:
            return float(stream.duration * stream.time_base)
        return container.duration / 1_000_000 if container.duration else 0.0
    finally:
        container.close()


def _evidence(frames: VideoFrames, t: float) -> list[EvidenceFrame]:
    """1-3 distinct sampled frames at t, t+1.5 s and t+3 s, with their real timestamps."""
    picked: list[int] = []
    for x in (t, t + 1.5, t + 3.0):
        i = frames.index_at(min(x, frames.duration))
        if i not in picked:
            picked.append(i)
    return [EvidenceFrame(timestamp_sec=round(frames.times[i], 1), image_base64=jpeg_b64(frames.images[i]))
            for i in sorted(picked)]


def demo_process(path: Path, job_id: str, *, delay_s: float = 1.0) -> JobResult:
    time.sleep(delay_s)  # lets the UI show its "processing" state, as a real job would
    duration = _probe_duration(path)
    fps = min(2.0, MAX_FRAMES / duration) if duration > 0 else 2.0
    frames = load_frames(path, fps)
    if not frames.times:
        raise ValueError("no video frames could be decoded")
    rules = _rules()
    violations = []
    for rule_id, pos, region in DEMO:
        rule = rules[rule_id]
        t = round(pos * frames.duration, 1)
        norm = rule["norm"]
        violations.append(Violation(
            type=TYPE_PREFIX + rule["type"],
            timecode_sec=t,
            description=(f"{DESC_PREFIX}Пример карточки нарушения «{rule['type']}» на {t:.1f} с. "
                         f"Кадры взяты из загруженного видео, но видео не анализировалось. "
                         f"Норматив: {norm['act']}, {norm['point']}."),
            evidence_frames=_evidence(frames, t),
            rule_id=rule_id,
            rule=RuleRef(act=norm["act"], point=norm["point"], strength=rule["strength"], url=norm.get("url")),
            confidence=0.0,
            region=list(region),
        ))
    return JobResult(job_id=job_id, status="done", verdict="violation", violations=violations)
