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
    return sorted(picked[:3])


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
            text = "Возможное несоответствие: " + text[:1].lower() + text[1:]
        text += f" Норматив: {rule.norm['act']}, {rule.norm['point']}."
        frames_b64 = [EvidenceFrame(timestamp_sec=x, image_base64=jpeg_b64(frames.at(x))) for x in _evidence_times(t, te, frames.duration)]
        out.append(Violation(type=rule.type, timecode_sec=t, description=text, evidence_frames=frames_b64,
                             rule_id=rule.id, confidence=round(score, 3),
                             rule=RuleRef(act=rule.norm["act"], point=rule.norm["point"], strength=rule.strength, url=rule.norm.get("url"))))
    return out
