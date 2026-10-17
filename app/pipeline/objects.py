"""S2b: open-vocabulary object boxes (OWLv2) on key frames.

OWLv2 says WHERE an object is; Qwen still decides WHAT happens. Object events only add probe times and evidence
boxes: a detection alone never creates a violation (threshold is low on purpose, Qwen filters).
"""
import bisect
import time
from dataclasses import dataclass

import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import Box, center, clip, contains, expand
from app.pipeline.people import PeopleTimeline, Track

QUERIES = {"stack of banknotes": "cash", "cash money": "cash", "mobile phone": "phone", "safe": "safe",
           "bag": "bag", "keys": "keys", "laptop": "laptop", "door": "door", "cup": "cup"}   # text query -> label
MAX_PER_LABEL = 5     # per frame and label, highest scores first: a global top-N let laptops, reflections and the
                      # landline push out cash in hands (0.14, the weakest true hit)
MAX_FRAMES = 200      # key frames per video
MIN_GAP_S = 0.4       # extra key frames closer than this to a chosen one add nothing
DESK_K, PHONE_K, SAFE_K = 1.2, 1.2, 1.5   # zone / person box expansion for the derived events
DEBOUNCE_N = 2        # cash on desk: a state change must hold for this many consecutive key frames
PHONE_MIN = 0.2       # the desk landline next to the cashier scores 0.13-0.18 in most frames (CPU check 17.10)


@dataclass(frozen=True)
class Det:
    label: str
    score: float
    box: Box              # normalized x1, y1, x2, y2


@dataclass
class ObjectEvent:
    t: float
    kind: str             # cash_on_desk_appeared | cash_on_desk_disappeared | phone_near_cashier | keys_near_safe
    score: float
    box: Box


class OwlDetector:
    """google/owlv2-base-patch16-ensemble (Apache-2.0) via transformers: fp16 on cuda, fp32 on cpu."""

    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble", device: str = "cpu",
                 threshold: float = 0.13, queries: dict[str, str] | None = None, nms_iou: float = 0.5):
        import torch
        from torchvision.ops import batched_nms
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        queries = dict(queries or QUERIES)
        self.labels = sorted(set(queries.values()))
        self.torch, self.nms, self.device, self.threshold, self.nms_iou = torch, batched_nms, device, threshold, nms_iou
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).eval().to(device=device, dtype=self.dtype)
        text = self.processor(text=[list(queries)], return_tensors="pt")      # one image's queries: (Q, L)
        self.ids, self.mask = text["input_ids"].to(device), text["attention_mask"].to(device)
        self.group = torch.tensor([self.labels.index(g) for g in queries.values()], device=device)

    def detect(self, images: list[np.ndarray]) -> list[list[Det]]:
        if not images:
            return []
        torch, n = self.torch, len(images)
        pix = self.processor.image_processor(images=list(images), return_tensors="pt", device=self.device)["pixel_values"]
        with torch.inference_mode():
            out = self.model(input_ids=self.ids.repeat(n, 1), attention_mask=self.mask.repeat(n, 1),
                             pixel_values=pix.to(device=self.device, dtype=self.dtype))
            scores, query = out.logits.float().max(-1)                        # best query per predicted box
            scores = scores.sigmoid()
            cx, cy, bw, bh = out.pred_boxes.float().unbind(-1)                 # relative to the padded square
            boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
            result = []
            for im, s, q, b in zip(images, scores, query, boxes):
                keep = s >= self.threshold
                s, g, b = s[keep], self.group[q[keep]], b[keep]
                keep = self.nms(b, s, g, self.nms_iou)                         # decreasing score
                h, w = im.shape[:2]
                sx, sy = max(h, w) / w, max(h, w) / h                         # OWLv2 pads bottom/right to a square
                dets, per_label = [], dict.fromkeys(self.labels, 0)
                for si, gi, bi in zip(s[keep].cpu(), g[keep].cpu(), b[keep].cpu()):
                    label, box = self.labels[int(gi)], tuple(float(v) * k for v, k in zip(bi, (sx, sy, sx, sy)))
                    if per_label[label] >= MAX_PER_LABEL or center(box)[0] > 1 or center(box)[1] > 1:
                        continue                                               # centre in the padding: not an object
                    per_label[label] += 1
                    dets.append(Det(label, round(float(si), 3), clip(tuple(round(v, 4) for v in box))))
                result.append(dets)
        return result


class FakeObjectDetector:
    """Test double: fn(image) -> list[Det] for each image."""

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def detect(self, images: list[np.ndarray]) -> list[list[Det]]:
        self.calls += len(images)
        return [list(self.fn(im)) for im in images]


@dataclass
class ObjectTimeline:
    times: list[float]
    dets: list[list[Det]]

    def best(self, label: str, t: float, tol: float = 1.0, within: Box | None = None) -> Det | None:
        """Highest-scoring `label` detection within +-tol s of t (centre inside `within` when given)."""
        i0 = bisect.bisect_left(self.times, t - tol - 1e-6)
        i1 = bisect.bisect_right(self.times, t + tol + 1e-6)
        return max((d for ds in self.dets[i0:i1] for d in ds if d.label == label
                    and (within is None or contains(within, *center(d.box)))), key=lambda d: d.score, default=None)


def key_times(frames: VideoFrames, fps: float, extra: list[float], cap: int = MAX_FRAMES) -> list[float]:
    """Decoded-frame times for the detector: a 1/fps grid plus the frames nearest to the extra times (people events,
    zone changes) not within MIN_GAP_S of a chosen frame; extras are strided to fit the cap."""
    snap = lambda t: frames.times[frames.index_at(t)]                        # noqa: E731
    grid = sorted({snap(i / fps) for i in range(int(frames.duration * fps) + 1)})
    if len(grid) > cap:
        grid = grid[::-(-len(grid) // cap)]
    chosen, picked = list(grid), []
    for t in sorted({snap(t) for t in extra if 0 <= t <= frames.duration}):
        i = bisect.bisect_left(chosen, t)
        if all(abs(chosen[j] - t) >= MIN_GAP_S - 1e-6 for j in (i - 1, i) if 0 <= j < len(chosen)):
            bisect.insort(chosen, t)
            picked.append(t)
    budget = cap - len(grid)
    if len(picked) > budget:
        picked = [picked[int(i * len(picked) / budget)] for i in range(budget)] if budget > 0 else []
    return sorted(grid + picked)


def detect_timeline(frames: VideoFrames, detector, times: list[float], batch: int = 8,
                    deadline: float | None = None) -> tuple[ObjectTimeline, bool]:
    """Run the detector on frames at `times` in batches; past `deadline` (time.monotonic) stop -> (partial, True)."""
    out_t: list[float] = []
    out_d: list[list[Det]] = []
    for i in range(0, len(times), batch):
        if i and deadline is not None and time.monotonic() > deadline:
            return ObjectTimeline(out_t, out_d), True
        chunk = times[i:i + batch]
        out_d += detector.detect([frames.at(t) for t in chunk])
        out_t += chunk
    return ObjectTimeline(out_t, out_d), False


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _best(dets: list[Det], label: str, ok) -> Det | None:
    return max((d for d in dets if d.label == label and ok(d.box)), key=lambda d: d.score, default=None)


def _changes(flags: list[bool], n: int = DEBOUNCE_N) -> list[tuple[int, bool]]:
    """(first index, new state) of each run of >= n equal flags that differs from the current state; the first
    such run only sets the initial state (cash already on the desk at the start is not an event)."""
    out, state, i = [], None, 0
    while i < len(flags):
        j = i
        while j + 1 < len(flags) and flags[j + 1] == flags[i]:
            j += 1
        if j - i + 1 >= n:
            if state is not None and flags[i] != state:
                out.append((i, flags[i]))
            state = flags[i]
        i = j + 1
    return out


def _starts(times: list[float], hits: list[Det | None], kind: str) -> list[ObjectEvent]:
    return [ObjectEvent(times[i], kind, d.score, d.box) for i, d in enumerate(hits)
            if d is not None and (i == 0 or hits[i - 1] is None)]


def object_events(tl: ObjectTimeline, desk: Box | None, safe: Box | None, people: PeopleTimeline,
                  tracks: list[Track]) -> list[ObjectEvent]:
    """cash_on_desk appeared/disappeared (debounced), phone near the cashier's box, keys near the safe."""
    events: list[ObjectEvent] = []
    if desk is not None:
        near = expand(desk, DESK_K)
        cash = [_best(ds, "cash", lambda b: contains(near, *center(b))) for ds in tl.dets]
        for i, present in _changes([d is not None for d in cash]):
            ref = cash[i] if present else next(d for d in reversed(cash[:i]) if d is not None)
            events.append(ObjectEvent(tl.times[i], "cash_on_desk_" + ("appeared" if present else "disappeared"),
                                      ref.score, ref.box))
    cashier = next((tr for tr in tracks if tr.id == people.cashier_id), None)
    if cashier is not None:
        phone = []
        for t, ds in zip(tl.times, tl.dets):
            pb = cashier.box_at(t)
            phone.append(_best([d for d in ds if d.score >= PHONE_MIN], "phone",
                               lambda b: _overlaps(b, expand(pb, PHONE_K))) if pb else None)
        events += _starts(tl.times, phone, "phone_near_cashier")
    if safe is not None:
        near_safe = expand(safe, SAFE_K)
        events += _starts(tl.times, [_best(ds, "keys", lambda b: contains(near_safe, *center(b))) for ds in tl.dets],
                          "keys_near_safe")
    return sorted(events, key=lambda e: e.t)
