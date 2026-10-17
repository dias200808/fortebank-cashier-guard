import numpy as np

from app.pipeline.frames import apply_rotation, load_frames


def ramp(i):
    return np.full((160, 96, 3), min(255, i * 3), np.uint8)


def test_samples_at_requested_fps(tmp_video):
    vf = load_frames(tmp_video(90, 30, ramp), fps=5.0)
    assert 14 <= len(vf.times) <= 16
    assert abs(vf.times[1] - vf.times[0] - 0.2) < 0.05
    assert vf.brightness[-1] > vf.brightness[0]
    assert 2.8 <= vf.duration <= 3.1


def test_at_returns_nearest_frame(tmp_video):
    vf = load_frames(tmp_video(90, 30, ramp), fps=5.0)
    assert abs(vf.at(2.0).mean() - 180) < 25      # frame ~60 -> 180
    assert vf.index_at(-1) == 0 and vf.index_at(99) == len(vf.times) - 1


def test_timestamps_start_at_zero_with_offset_stream(tmp_video):
    vf = load_frames(tmp_video(30, 30, ramp), fps=5.0)
    assert vf.times[0] < 0.05


def test_apply_rotation_clockwise():
    img = np.zeros((4, 2, 3), np.uint8)
    img[0, 0] = 255                       # marker top-left
    out = apply_rotation(img, -90)        # display matrix -90 ccw = rotate clockwise
    assert out.shape[:2] == (2, 4) and out[0, -1, 0] == 255
