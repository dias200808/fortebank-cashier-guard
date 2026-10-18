"""S7: re-check each candidate on a short clip of zone crops with 2-3 phrasings, majority vote."""
import cv2
import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.people import PeopleTimeline
from app.pipeline.probes import Fact, zone_image
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


def build_items(c: Candidate, rule: Rule, frames: VideoFrames, zones: Zones, facts_by_id: dict[str, Fact],
                people: PeopleTimeline | None = None) -> list[tuple[list[np.ndarray], str]]:
    """zone "cashier": the tracked cashier's box per frame (frames with her out of view are dropped). scale: upscale
    factor for the crops of one question. In a 6-frame clip of the ~330x200 px desk crop Qwen misses a small stack of
    banknotes that it sees at 2x (X-26@11: 0.005/0.27 native, 0.88/1.0 at 2x; empty desks stay 0.0)."""
    default_zone = facts_by_id[c.fact].zone if c.fact in facts_by_id else "full"
    items = []
    for v in rule.verify:
        zone, ts = v.get("zone", default_zone), clip_times(c.t, v.get("window", [-3, 5]), frames.duration)
        imgs = [im for im in (zone_image(frames.at(x), zone, zones, x, people) for x in ts) if im is not None]
        imgs = imgs or [frames.at(x) for x in ts]                  # cashier never in view: full frames
        k = v.get("scale", 1)
        if k != 1:
            imgs = [cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) for im in imgs]
        items.append((imgs, v["q"] + CLIP_NOTE))
    return items


def decide(rule: Rule, probs: list[float]) -> tuple[bool, float]:
    """Majority vote; a question marked required: true is a veto (the context the rule rests on, e.g. the open safe
    of R22, must hold whatever the other phrasings say). A multiple-choice question's prob is already P(expected
    option)."""
    agree = [p if "options" in v or v.get("expect", "yes") == "yes" else 1.0 - p for v, p in zip(rule.verify, probs)]
    votes = sum(a > 0.5 for a in agree)
    vetoed = any(v.get("required") and a <= 0.5 for v, a in zip(rule.verify, agree))
    return votes * 2 > len(agree) and not vetoed, sum(agree) / len(agree)


def verify_all(cands: list[Candidate], rules_by_id: dict[str, Rule], frames: VideoFrames, zones: Zones,
               facts_by_id: dict[str, Fact], vlm: VLM,
               people: PeopleTimeline | None = None) -> list[tuple[Candidate, bool, float]]:
    items, owners, entries = [], [], []
    for i, c in enumerate(cands):
        rule = rules_by_id[c.rule_id]
        for v, it in zip(rule.verify, build_items(c, rule, frames, zones, facts_by_id, people)):
            items.append(it)
            owners.append(i)
            entries.append(v)
    yes = [k for k, v in enumerate(entries) if "options" not in v]
    probs = [0.0] * len(items)
    for k, p in zip(yes, vlm.yes_probs([items[k] for k in yes]) if yes else []):
        probs[k] = p
    for k, v in enumerate(entries):
        if "options" in v:                  # multiple choice: P(expected option); no answer counts as a "no"
            fn = getattr(vlm, "choice_probs", None)
            d = fn(*items[k], [str(o) for o in v["options"]]) if fn else None
            probs[k] = d.get(str(v["expect"]), 0.0) if d else 0.0
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
