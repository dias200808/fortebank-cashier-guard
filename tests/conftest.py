from pathlib import Path
from typing import Callable

import av
import numpy as np
import pytest

Painter = Callable[[int], np.ndarray]


def write_video(path: Path, n_frames: int, fps: int, painter: Painter, size: tuple[int, int] = (96, 160)) -> Path:
    """Write an mpeg4 .mov whose frame i is painter(i) (H x W x 3 uint8, W x H = size)."""
    w, h = size
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for i in range(n_frames):
        frame = av.VideoFrame.from_ndarray(painter(i), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture
def tmp_video(tmp_path):
    def make(n_frames: int, fps: int, painter: Painter, size=(96, 160), name="v.mov") -> Path:
        return write_video(tmp_path / name, n_frames, fps, painter, size)
    return make
