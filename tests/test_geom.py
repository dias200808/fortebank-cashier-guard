import numpy as np

from app.pipeline.geom import center, clip, contains, crop, expand, from_pixels


def test_clip_and_expand():
    assert clip((-0.1, 0.2, 1.3, 0.9)) == (0.0, 0.2, 1.0, 0.9)
    assert expand((0.4, 0.4, 0.6, 0.6), 2.0) == (0.3, 0.3, 0.7, 0.7)


def test_center_contains_from_pixels():
    b = from_pixels((10, 20, 30, 60), 100, 200)
    assert b == (0.1, 0.1, 0.3, 0.3)
    assert center(b) == (0.2, 0.2) and contains(b, 0.2, 0.2) and not contains(b, 0.5, 0.5)


def test_crop_none_returns_full_and_box_returns_region():
    img = np.zeros((200, 100, 3), np.uint8)
    assert crop(img, None) is img
    c = crop(img, (0.4, 0.4, 0.6, 0.6), margin=0.0)
    assert c.shape[:2] == (40, 20)
