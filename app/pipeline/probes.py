"""S4: narrow yes/no fact questions on zone crops -> P(yes) time series per fact."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import crop
from app.pipeline.people import PeopleTimeline
from app.pipeline.scene import Zones
from app.pipeline.vlm import VLM

Series = list[tuple[float, float]]
CASHIER_SHORT = 448   # px: the cashier's box is ~60-150 px wide, a few visual tokens; upscaled the VLM sees the hands
# a fact with `near` is read only by rules anchored on those S2 events (on_cashier_left: tol <= 2 s before the departure,
# post_s 5 s after, R04's dark bridge <= 4 s + 5 s before): asked only within NEAR_S of one. The other ~75 % of a long
# clip's probe frames skip it (X-4 on the production budget probed 184 of 240 frames before the 120 s stop).
NEAR_S = 10.0


@dataclass(frozen=True)
class Fact:
    id: str
    zone: str
    question: str
    near: tuple[str, ...] = ()        # S2 event kinds; empty = asked at every probe time


def load_facts(path: Path) -> list[Fact]:
    return [Fact(**{**d, "near": tuple(d.get("near") or ())}) for d in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


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


def zone_image(img: np.ndarray, zone: str, zones: Zones, t: float,
               people: PeopleTimeline | None = None) -> np.ndarray | None:
    """What a question about `zone` sees at time t. "cashier" = the tracked cashier's person box, tight (no margin: a
    client's hands across the desk stay out, X-4) and upscaled: on the full frame "is the cashier holding a phone" also
    fires on a client's phone (X-5). None = the cashier is out of view at t; with no tracked cashier, the full frame."""
    if zone != "cashier":
        return crop(img, zones.get(zone))
    if people is None or people.cashier_id is None:
        return img
    box = people.cashier_box_at(t)
    if box is None:
        return None
    im = crop(img, box, margin=0.0)
    if min(im.shape[:2]) == 0:
        return None                                # box at the frame edge: nothing of her left in the crop
    k = CASHIER_SHORT / min(im.shape[:2])
    return cv2.resize(im, (round(im.shape[1] * k), round(im.shape[0] * k)), interpolation=cv2.INTER_CUBIC) if k > 1 else im


def run_probes(frames: VideoFrames, zones: Zones, vlm: VLM, facts: list[Fact], times: list[float],
               people: PeopleTimeline | None = None) -> dict[str, Series]:
    """P(yes) per fact and time; a cashier-zone question while the cashier is out of view is P = 0, not asked. A fact
    with `near` gets no sample at all farther than NEAR_S from its events (without S2 people: asked everywhere)."""
    items, keys = [], []
    series: dict[str, Series] = {f.id: [] for f in facts}
    anchors = {f.id: [e.t for e in people.events if e.kind in f.near] for f in facts if f.near and people is not None}
    for t in times:
        img = frames.at(t)
        for f in facts:
            if f.id in anchors and not any(abs(t - a) <= NEAR_S for a in anchors[f.id]):
                continue
            im = zone_image(img, f.zone, zones, t, people)
            if im is None:
                series[f.id].append((t, 0.0))
                continue
            items.append(([im], f.question))
            keys.append((f.id, t))
    for (fid, t), p in zip(keys, vlm.yes_probs(items)):
        series[fid].append((t, p))
    return {fid: sorted(pts) for fid, pts in series.items()}
