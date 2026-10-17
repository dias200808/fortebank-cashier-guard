import av
import numpy as np


def test_write_video_roundtrip(tmp_video):
    path = tmp_video(30, 30, lambda i: np.full((160, 96, 3), 128, np.uint8))
    c = av.open(str(path))
    frames = list(c.decode(c.streams.video[0]))
    assert len(frames) == 30
    assert abs(frames[0].to_ndarray(format="rgb24").mean() - 128) < 10
