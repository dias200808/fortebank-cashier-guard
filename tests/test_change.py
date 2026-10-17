import numpy as np

from app.pipeline.change import change_times, dense_times, zone_change
from app.pipeline.frames import VideoFrames
from app.pipeline.people import Track

DESK = (0.0, 0.5, 1.0, 1.0)


def frames(square_from=None, n=10):
    imgs = []
    for i in range(n):
        im = np.zeros((100, 100, 3), np.uint8)
        if square_from is not None and i >= square_from:
            im[60:80, 40:60] = 255
        imgs.append(im)
    return VideoFrames([i * 0.2 for i in range(n)], imgs, n * 0.2)


def test_object_appearing_on_desk_spikes_once():
    s = zone_change(frames(square_from=5), DESK, [])
    spikes = change_times(s, 0.02)
    assert spikes == [1.0]


def test_change_hidden_by_person_is_masked_but_visible_as_hand_activity():
    person = Track(1, [i * 0.2 for i in range(10)], [(0.3, 0.5, 0.7, 0.9)] * 10)
    vf = frames(square_from=5)
    assert change_times(zone_change(vf, DESK, [person], mask_people=True), 0.02) == []
    assert change_times(zone_change(vf, DESK, [person], mask_people=False), 0.02) == [1.0]


def test_no_zone_returns_empty():
    assert zone_change(frames(), None, []) == []


def test_dense_times():
    assert dense_times([1.0]) == [0.6, 0.8, 1.0, 1.2, 1.4]


def test_moving_person_does_not_create_scene_change():
    imgs, boxes, times = [], [], []
    for i in range(10):
        im = np.zeros((100, 100, 3), np.uint8)
        x = 10 + 8 * i
        im[55:95, x:x + 20] = 200                       # person body moving right inside the desk zone
        imgs.append(im)
        times.append(i * 0.2)
        boxes.append((x / 100, 0.55, (x + 20) / 100, 0.95))
    person = Track(1, times, boxes)
    vf = VideoFrames(times, imgs, 2.0)
    assert change_times(zone_change(vf, DESK, [person], mask_people=True), 0.02) == []
