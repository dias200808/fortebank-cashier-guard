"""S7: re-check each candidate on a short clip of zone crops with 2-3 phrasings, majority vote."""
import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import crop
from app.pipeline.probes import Fact
from app.pipeline.rules import Candidate, Rule
from app.pipeline.scene import Zones
from app.pipeline.vlm import VLM

CLIP_NOTE = " The images are consecutive video frames in time order."


def clip_times(t: float, window: list[float], duration: float, step: float = 1.0, max_frames: int = 8,
               min_frames: int = 6) -> list[float]:
    a, b = max(0.0, t + window[0]), min(duration, t + window[1])
    if b > a:
        step = min(step, (b - a) / (min_frames - 1))
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
