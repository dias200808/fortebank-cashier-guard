import numpy as np

from app.pipeline.scene import DEFAULT_DESK, Zones, locate_zones, parse_bbox
from app.pipeline.vlm import FakeVLM, VLMError

IMG = np.zeros((848, 464, 3), np.uint8)


def test_parse_rel1000_and_abs():
    assert parse_bbox('{"bbox_2d": [100, 200, 500, 600]}', 464, 848, "rel1000") == (0.1, 0.2, 0.5, 0.6)
    assert parse_bbox('{"bbox_2d": [46.4, 84.8, 232, 424]}', 464, 848, "abs") == (0.1, 0.1, 0.5, 0.5)


def test_parse_rejects_null_and_tiny():
    assert parse_bbox('{"bbox_2d": null}', 464, 848, "rel1000") is None
    assert parse_bbox('[10, 10, 12, 12]', 464, 848, "rel1000") is None
    assert parse_bbox("no box here", 464, 848, "rel1000") is None


def test_locate_zones_uses_vlm_and_defaults_desk():
    def reply(images, prompt):
        return '{"bbox_2d": [600, 300, 900, 600]}' if "safe" in prompt else '{"bbox_2d": null}'
    z = locate_zones(IMG, FakeVLM(reply=reply))
    assert z.safe == (0.6, 0.3, 0.9, 0.6) and z.door is None and z.desk == DEFAULT_DESK
    assert z.get("full") is None and z.get("safe") == z.safe


def test_locate_zones_survives_vlm_error():
    def reply(images, prompt):
        raise VLMError("down")
    z = locate_zones(IMG, FakeVLM(reply=reply))
    assert z == Zones(desk=DEFAULT_DESK)


def test_parse_rejects_malformed_numbers_without_crashing():
    assert parse_bbox("[1.2.3, 1, 2, 3]", 464, 848, "rel1000") is None
    assert parse_bbox("[..., 1, 2, 3]", 464, 848, "rel1000") is None
    assert parse_bbox("[x1, y1, x2, y2]", 464, 848, "rel1000") is None


def test_locate_zones_stops_after_first_vlm_error():
    def reply(images, prompt):
        raise VLMError("down")
    vlm = FakeVLM(reply=reply)
    assert locate_zones(IMG, vlm) == Zones(desk=DEFAULT_DESK) and vlm.calls == 1
