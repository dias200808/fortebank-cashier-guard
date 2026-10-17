"""S4: narrow yes/no fact questions on zone crops -> P(yes) time series per fact."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import crop
from app.pipeline.scene import Zones
from app.pipeline.vlm import VLM

Series = list[tuple[float, float]]


@dataclass(frozen=True)
class Fact:
    id: str
    zone: str
    question: str


def load_facts(path: Path) -> list[Fact]:
    return [Fact(**d) for d in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


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


def run_probes(frames: VideoFrames, zones: Zones, vlm: VLM, facts: list[Fact], times: list[float]) -> dict[str, Series]:
    items, keys = [], []
    for t in times:
        img = frames.at(t)
        for f in facts:
            items.append(([crop(img, zones.get(f.zone))], f.question))
            keys.append((f.id, t))
    series: dict[str, Series] = {f.id: [] for f in facts}
    for (fid, t), p in zip(keys, vlm.yes_probs(items)):
        series[fid].append((t, p))
    return series
