import base64

import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.report import build_violations, jpeg_b64
from app.pipeline.rules import Candidate, Rule

VF = VideoFrames([i * 0.2 for i in range(100)], [np.full((60, 40, 3), 100, np.uint8)] * 100, 20.0)
R01 = Rule("R01", "Наличные без присмотра", ["on_cashier_left"], ["F_cash_desk"], {"offset_s": 0.0},
           {"act": "Правила № 231", "point": "п. 20 ч. 2", "url": "u"}, "В", "В {t} с кассир ушёл (кадр {te} с).")
R08 = Rule("R08", "Дверь открыта", ["duration"], ["F_door_open"], {}, {"act": "Правила № 14", "point": "п. 16"}, "Н", "С {t} с дверь открыта.")


def test_jpeg_b64_is_jpeg():
    assert base64.b64decode(jpeg_b64(np.zeros((10, 10, 3), np.uint8)))[:2] == b"\xff\xd8"


def test_build_violation_fields_and_frames():
    v = build_violations([(Candidate("R01", 9.4, 10.2, "F_cash_desk", 0.9), 0.85)], {"R01": R01}, VF)
    assert len(v) == 1
    assert v[0].type == "Наличные без присмотра" and v[0].timecode_sec == 9.4
    assert "9.4" in v[0].description and "Правила № 231, п. 20 ч. 2" in v[0].description
    assert 1 <= len(v[0].evidence_frames) <= 3 and v[0].evidence_frames[0].timestamp_sec == 9.4
    assert v[0].rule.point == "п. 20 ч. 2"


def test_merge_same_rule_within_window_keeps_best():
    a = (Candidate("R01", 9.0, 9.0, "F_cash_desk", 0.7), 0.6)
    b = (Candidate("R01", 11.0, 11.0, "F_cash_desk", 0.9), 0.9)
    c = (Candidate("R01", 18.0, 18.0, "F_cash_desk", 0.9), 0.8)
    v = build_violations([a, b, c], {"R01": R01}, VF)
    assert [x.timecode_sec for x in v] == [11.0, 18.0]


def test_indirect_norm_is_phrased_as_possible():
    v = build_violations([(Candidate("R08", 6.0, 8.0, "F_door_open", 0.9), 0.9)], {"R08": R08}, VF)
    assert v[0].description.startswith("Возможное несоответствие")


def test_evidence_frames_sorted_chronologically():
    v = build_violations([(Candidate("R01", 9.4, 7.0, "F_cash_desk", 0.9), 0.85)], {"R01": R01}, VF)
    ts = [e.timestamp_sec for e in v[0].evidence_frames]
    assert len(ts) >= 2 and ts == sorted(ts)


def test_hedge_prefix_lowercases_template_start():
    v = build_violations([(Candidate("R08", 6.0, 8.0, "F_door_open", 0.9), 0.9)], {"R08": R08}, VF)
    assert v[0].description.startswith("Возможное несоответствие: с ")
