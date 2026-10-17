"""S9: merge accepted candidates, pick evidence frames, write descriptions from templates."""
import base64
import io

import numpy as np
from PIL import Image

from app.pipeline.frames import VideoFrames
from app.pipeline.probes import Fact
from app.pipeline.rules import Candidate, Rule
from app.pipeline.scene import DEFAULT_DESK, Zones
from app.schemas import EvidenceFrame, RuleRef, Violation

NAMED_ZONES = ("desk", "safe", "door", "laptop")
FAMILY_ZONES = {"door_left_open": "door", "cashier_left_with_client": "desk"}  # fact-less rules watched in one zone


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


def region_for(rule: Rule, facts_by_id: dict[str, Fact], zones: Zones | None) -> list[float] | None:
    """Box of the rule's primary fact: the first fact whose zone is a named zone located on this video; for a
    fact-less rule, the zone its family watches (R08 door, R05 desk). None for full-frame-only rules,
    people-only rules and zones S1 could not find (the default desk is a guess)."""
    if zones is None:
        return None
    names = [f.zone for f in (facts_by_id.get(fid) for fid in rule.facts) if f is not None]
    names += [FAMILY_ZONES[fam] for fam in rule.families if fam in FAMILY_ZONES]
    for name in names:
        box = zones.get(name) if name in NAMED_ZONES else None
        if box is not None and not (name == "desk" and tuple(box) == DEFAULT_DESK):
            return [round(float(v), 3) for v in box]
    return None


def build_violations(accepted: list[tuple[Candidate, float]], rules_by_id: dict[str, Rule],
                     frames: VideoFrames, merge_s: float = 5.0, *, zones: Zones | None = None,
                     facts_by_id: dict[str, Fact] | None = None) -> list[Violation]:
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
                             rule_id=rule.id, confidence=round(score, 3), region=region_for(rule, facts_by_id or {}, zones),
                             rule=RuleRef(act=rule.norm["act"], point=rule.norm["point"], strength=rule.strength, url=rule.norm.get("url"))))
    return out
