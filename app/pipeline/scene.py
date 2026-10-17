"""S1: locate desk, safe, door and laptop on this video's own frame (no hard-coded zones)."""
import re
from dataclasses import dataclass

import numpy as np

from app.pipeline.geom import Box, clip
from app.pipeline.vlm import VLM, VLMError

OBJECTS = {
    "desk": "the cashier's work desk (table with a laptop and a banknote counting machine)",
    "safe": "the metal safe (small grey cabinet with a door)",
    "door": "the entrance door of the room",
    "laptop": "the laptop computer screen",
}
PROMPT = ('Locate {desc} in the image. Reply with JSON only, like {{"bbox_2d": [x1, y1, x2, y2]}}. '
          'If it is not visible, reply {{"bbox_2d": null}}.')
DEFAULT_DESK: Box = (0.0, 0.3, 1.0, 0.8)
_NUM = r"-?\d+(?:\.\d+)?"
_NUM4 = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]")


@dataclass
class Zones:
    desk: Box | None = None
    safe: Box | None = None
    door: Box | None = None
    laptop: Box | None = None

    def get(self, name: str) -> Box | None:
        return None if name == "full" else getattr(self, name, None)


def parse_bbox(text: str, width: int, height: int, mode: str) -> Box | None:
    m = _NUM4.search(text or "")
    if not m:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in m.groups())
    except ValueError:
        return None
    sx, sy = (1000.0, 1000.0) if mode == "rel1000" else (float(width), float(height))
    b = clip((round(x1 / sx, 4), round(y1 / sy, 4), round(x2 / sx, 4), round(y2 / sy, 4)))
    if b[2] - b[0] < 0.02 or b[3] - b[1] < 0.02:
        return None
    return b


def locate_zones(image: np.ndarray, vlm: VLM, mode: str = "rel1000") -> Zones:
    h, w = image.shape[:2]
    zones = Zones()
    for name, desc in OBJECTS.items():
        try:
            setattr(zones, name, parse_bbox(vlm.text([image], PROMPT.format(desc=desc), 64), w, h, mode))
        except VLMError:
            break                      # server down: don't pay the retry/timeout cost for every object
    if zones.desk is None:
        zones.desk = DEFAULT_DESK
    return zones
