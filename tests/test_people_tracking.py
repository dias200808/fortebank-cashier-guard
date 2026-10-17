import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.people import Track, track_people


class ScriptedDetector:
    """Returns one person box per frame whose x position is encoded in the frame's red channel."""

    def __call__(self, images):
        out = []
        for im in images:
            x = int(im[0, 0, 0])
            out.append(np.array([[x, 50, x + 20, 120, 0.9]], np.float32) if x > 0 else np.zeros((0, 5), np.float32))
        return out


def frames_with_person(xs):
    imgs = []
    for x in xs:
        im = np.zeros((160, 96, 3), np.uint8)
        im[0, 0, 0] = x
        imgs.append(im)
    return VideoFrames([round(i * 0.2, 3) for i in range(len(xs))], imgs, len(xs) * 0.2)


def test_track_box_at_and_height():
    tr = Track(1, [0.0, 0.2, 0.4], [(0.1, 0.1, 0.2, 0.5)] * 3)
    assert tr.box_at(0.21) == (0.1, 0.1, 0.2, 0.5) and tr.box_at(1.0) is None
    assert abs(tr.height() - 0.4) < 1e-9 and tr.start == 0.0 and tr.end == 0.4


def test_single_person_gives_one_track():
    tracks = track_people(frames_with_person([30] * 20), ScriptedDetector(), fps=5.0)
    assert len(tracks) == 1 and len(tracks[0].times) >= 17


def test_empty_video_gives_no_tracks():
    assert track_people(frames_with_person([0] * 10), ScriptedDetector(), fps=5.0) == []


def test_short_detection_gap_keeps_one_track():
    tracks = track_people(frames_with_person([30] * 10 + [0] * 3 + [30] * 10), ScriptedDetector(), fps=5.0)
    assert len(tracks) == 1


def test_box_at_nearest_and_tolerance_boundary():
    b0, b1, b2 = (0.1, 0.1, 0.2, 0.2), (0.3, 0.3, 0.4, 0.4), (0.5, 0.5, 0.6, 0.6)
    tr = Track(1, [0.0, 0.2, 0.4], [b0, b1, b2])
    assert tr.box_at(0.29) == b1
    assert tr.box_at(0.31) == b2
    assert tr.box_at(0.65) == b2
    assert tr.box_at(0.66) is None


def test_detector_uses_native_input_size():
    from pathlib import Path

    import pytest

    torch = pytest.importorskip("torch")
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"
    if not ckpt.exists():
        pytest.skip("detector weights not cached")
    from app.pipeline.people import TorchvisionPersonDetector

    det = TorchvisionPersonDetector("frcnn_mobile", "cpu")
    assert det.model.transform.min_size == (480,) and det.model.transform.max_size == 864
