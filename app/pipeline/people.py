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
LOST_TRACK_S = 2.0  # seconds a lost person is kept alive


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

    def __init__(self, kind: str = "frcnn_mobile", device: str = "cpu", score: float = 0.5,
                 min_size: int = 480, max_size: int = 864):
        import torch
        import torchvision.models.detection as tvd

        # ~native size for 464x848 frames (torchvision's default 800/1333 upscales ~1.6x for no gain)
        kw = dict(box_score_thresh=0.3, min_size=min_size, max_size=max_size)
        if kind == "frcnn_r50v2":
            model = tvd.fasterrcnn_resnet50_fpn_v2(weights=tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, **kw)
        else:
            model = tvd.fasterrcnn_mobilenet_v3_large_fpn(weights=tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, **kw)
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
    # trackers counts lost_track_buffer in 30-fps frames and rescales by frame_rate.
    tracker = trackers.ByteTrackTracker(frame_rate=fps, lost_track_buffer=int(LOST_TRACK_S * 30), track_activation_threshold=0.5)
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


H_REL = 0.5        # glass reflections are < half the cashier's height (probe 2026-10-17)
AWAY_D = 0.05      # away = centre > 5 % of frame height from the seat
AWAY_S = 0.6       # ... for at least 0.6 s
CHAIN_GAP_S = 3.0  # away runs closer than this belong to one departure
EDGE_S = 1.0       # tracks starting/ending within 1 s of the clip edge are not entries/exits
MIN_VISIT_S = 2.0  # non-cashier tracks shorter than this are door/glass artefacts (X-26, X-22 probe 2026-10-17)
SEAT_LINK_D = 0.08  # a tracker ID switch at the seat: a track centred this close to the seat is the same cashier
ABSENT_S = 2.0      # cashier out of view this long inside a departure chain counts as leaving, even if he returns


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


def _conflicts(cand: Track, base: Track) -> bool:
    """Shared timestamps inside base's interior mean two people at once; 0.4 s at the ends is handover."""
    shared = set(cand.times) & set(base.times)
    lo, hi = base.start + 0.4, base.end - 0.4
    return any(lo <= t <= hi for t in shared)


def _seat_linked(cashier: Track, tracks: list[Track], sx: float, sy: float, aspect: float) -> tuple[Track, set[int]]:
    """Merge tracks that sit on the cashier's seat and never coexist with the cashier (tracker ID switches)."""
    merged, absorbed = cashier, set()
    for cand in sorted((tr for tr in tracks if tr.id != cashier.id), key=lambda tr: tr.start):
        cx = st.median(center(b)[0] for b in cand.boxes)
        cy = st.median(center(b)[1] for b in cand.boxes)
        if ((cx - sx) * aspect) ** 2 + (cy - sy) ** 2 > SEAT_LINK_D ** 2 or _conflicts(cand, merged):
            continue
        pairs = sorted(zip(merged.times + cand.times, merged.boxes + cand.boxes), key=lambda p: p[0])
        merged = Track(cashier.id, [t for t, _ in pairs], [b for _, b in pairs])
        absorbed.add(cand.id)
    return merged, absorbed


def analyze_people(tracks: list[Track], times: list[float], desk: Box | None, safe: Box | None,
                   duration: float, aspect: float) -> PeopleTimeline:
    if not tracks or not times:
        return PeopleTimeline(times, None, [True] * len(times), [0] * len(times), [])
    near_desk = expand(desk, 1.4) if desk else None

    def desk_score(tr: Track) -> tuple[int, int]:
        hits = sum(contains(near_desk, *center(b)) for b in tr.boxes) if near_desk else len(tr.boxes)
        return hits, len(tr.boxes)

    primary = max(tracks, key=desk_score)
    sx = st.median(center(b)[0] for b in primary.boxes)
    sy = st.median(center(b)[1] for b in primary.boxes)
    cashier, absorbed = _seat_linked(primary, tracks, sx, sy, aspect)
    real = [tr for tr in tracks if tr.id not in absorbed
            and (tr.id == cashier.id or (tr.height() >= H_REL * cashier.height()
                                         and tr.end - tr.start >= MIN_VISIT_S))]
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
    absent_runs = _runs(times, [cashier.box_at(t) is None for t in times])

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
        out_of_view = any(a <= ra and rb <= b and times[rb] - times[ra] >= ABSENT_S for ra, rb in absent_runs)
        if (gone_for_good and cashier.end < duration - EDGE_S) or out_of_view:
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
