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
    series, prev, prev_mask = [], None, None
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
        # A person vacating a pixel between samples also changes it: exclude pixels masked in either sample.
        valid = mask & prev_mask if mask_people and prev_mask is not None else mask
        if prev is not None and valid.any():
            score = float((np.abs(g - prev) > diff_thr)[valid].mean())
        series.append((t, score))
        prev, prev_mask = g, mask
    return series


def change_times(series: list[tuple[float, float]], thr: float) -> list[float]:
    return [t for t, s in series if s > thr]


def dense_times(change_ts: list[float], step: float = 0.2, span: float = 0.4) -> list[float]:
    n = int(round(span / step))
    return sorted({round(t + k * step, 2) for t in change_ts for k in range(-n, n + 1) if t + k * step >= 0})
