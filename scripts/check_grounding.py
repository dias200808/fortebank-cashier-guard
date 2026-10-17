"""Find which bbox convention the served VLM uses: draws a red box at a known place and compares."""
import sys

import numpy as np

from app.pipeline.scene import parse_bbox
from app.pipeline.vlm import OpenAIVLM

url, model = sys.argv[1], sys.argv[2]
img = np.full((848, 464, 3), 200, np.uint8)
img[424:636, 232:406] = (220, 30, 30)                       # truth (0.5, 0.5, 0.875, 0.75)
truth = (0.5, 0.5, 0.875, 0.75)
text = OpenAIVLM(url, model).text([img], 'Locate the red rectangle. Reply with JSON only, like {"bbox_2d": [x1, y1, x2, y2]}.', 64)
print("raw:", text)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


for mode in ("rel1000", "abs"):
    b = parse_bbox(text, 464, 848, mode)
    print(mode, b, "IoU", round(iou(b, truth), 2) if b else None)
