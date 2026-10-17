"""S0: decode a video into upright RGB frames sampled at a fixed rate, timestamps from stream start."""
import bisect
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass
class VideoFrames:
    times: list[float]
    images: list[np.ndarray]
    duration: float
    brightness: list[float] = field(default_factory=list)
    sharpness: list[float] = field(default_factory=list)

    def index_at(self, t: float) -> int:
        i = bisect.bisect_left(self.times, t)
        if i <= 0:
            return 0
        if i >= len(self.times):
            return len(self.times) - 1
        return i if self.times[i] - t < t - self.times[i - 1] else i - 1

    def at(self, t: float) -> np.ndarray:
        return self.images[self.index_at(t)]

    def times_between(self, t0: float, t1: float) -> list[float]:
        return [t for t in self.times if t0 <= t <= t1]


def apply_rotation(img: np.ndarray, degrees_ccw: float) -> np.ndarray:
    """PyAV frame.rotation is the display-matrix angle (counter-clockwise); np.rot90 is counter-clockwise too."""
    k = int(round(degrees_ccw / 90)) % 4
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def load_frames(path: Path, fps: float) -> VideoFrames:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        start = stream.start_time or 0
        times, images, bright, sharp = [], [], [], []
        next_t = 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float((frame.pts - start) * stream.time_base)
            if t + 1e-6 < next_t:
                continue
            while next_t <= t + 1e-6:
                next_t += 1.0 / fps
            img = apply_rotation(frame.to_ndarray(format="rgb24"), getattr(frame, "rotation", 0) or 0)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            times.append(round(max(0.0, t), 3))
            images.append(img)
            bright.append(float(gray.mean()))
            sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        stream_dur = float(stream.duration * stream.time_base) if stream.duration else 0.0
        duration = max(times[-1] if times else 0.0, stream_dur)
        return VideoFrames(times, images, duration, bright, sharp)
    finally:
        container.close()
