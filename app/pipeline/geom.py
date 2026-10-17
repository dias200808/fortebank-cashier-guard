"""Normalized box helpers. A Box is (x1, y1, x2, y2) in [0, 1] relative to the frame."""
import numpy as np

Box = tuple[float, float, float, float]


def clip(b: Box) -> Box:
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in b)
    return (x1, y1, x2, y2)


def expand(b: Box, k: float) -> Box:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    hw, hh = (b[2] - b[0]) * k / 2, (b[3] - b[1]) * k / 2
    return clip((round(cx - hw, 6), round(cy - hh, 6), round(cx + hw, 6), round(cy + hh, 6)))


def center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def contains(b: Box, x: float, y: float) -> bool:
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def from_pixels(xyxy, w: int, h: int) -> Box:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return clip((x1 / w, y1 / h, x2 / w, y2 / h))


def crop(img: np.ndarray, b: Box | None, margin: float = 0.15) -> np.ndarray:
    if b is None:
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = expand(b, 1 + 2 * margin) if margin else b
    px1, py1 = int(round(x1 * w)), int(round(y1 * h))
    px2, py2 = max(px1 + 8, int(round(x2 * w))), max(py1 + 8, int(round(y2 * h)))
    return img[py1:min(h, py2), px1:min(w, px2)]
