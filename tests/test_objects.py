import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings, load_settings
from app.pipeline import run as run_mod
from app.pipeline.frames import VideoFrames
from app.pipeline.objects import (QUERIES, Det, FakeObjectDetector, ObjectTimeline, OwlDetector, detect_timeline,
                                  key_times, object_events)
from app.pipeline.people import PeopleTimeline, Track
from app.pipeline.probes import Fact, load_facts
from app.pipeline.report import build_violations, object_region
from app.pipeline.rules import Candidate, Rule, load_rules
from app.pipeline.run import Models, analyze, build_models
from app.pipeline.scene import Zones
from tests.test_run import SETTINGS, BlockDetector, cash_vlm, painter

DESK = (0.1, 0.5, 0.9, 0.8)
SAFE = (0.7, 0.1, 0.95, 0.4)
CASH = Det("cash", 0.4, (0.3, 0.6, 0.4, 0.7))
NO_PEOPLE = PeopleTimeline([], None, [], [], [])


def timeline(dets_per_t: list[list[Det]], step: float = 1.0) -> ObjectTimeline:
    return ObjectTimeline([round(i * step, 2) for i in range(len(dets_per_t))], dets_per_t)


def kinds(events):
    return [(e.t, e.kind) for e in events]


def test_cash_on_desk_appear_disappear_is_debounced():
    off_desk = Det("cash", 0.9, (0.0, 0.0, 0.05, 0.05))                 # centre outside the expanded desk
    dets = [[], [], [], [CASH], [CASH], [CASH], [CASH], [], [], [CASH], [], [off_desk], [off_desk]]
    ev = object_events(timeline(dets), DESK, None, NO_PEOPLE, [])
    assert kinds(ev) == [(3.0, "cash_on_desk_appeared"), (7.0, "cash_on_desk_disappeared")]
    assert ev[0].box == CASH.box and ev[1].box == CASH.box                # disappeared points at where it was


def test_cash_present_from_start_is_not_an_appearance():
    ev = object_events(timeline([[CASH], [CASH], [CASH], [], [], []]), DESK, None, NO_PEOPLE, [])
    assert kinds(ev) == [(3.0, "cash_on_desk_disappeared")]


def test_single_frame_blips_make_no_cash_events():
    ev = object_events(timeline([[], [], [CASH], [], [CASH], [], []]), DESK, None, NO_PEOPLE, [])
    assert ev == []


def test_phone_near_cashier_uses_the_cashier_track():
    times = [float(i) for i in range(6)]
    cashier = Track(1, times, [(0.3, 0.2, 0.6, 0.6)] * 6)
    visitor = Track(2, times, [(0.0, 0.0, 0.2, 0.3)] * 6)
    people = PeopleTimeline(times, 1, [False] * 6, [1] * 6, [])
    near = Det("phone", 0.3, (0.6, 0.3, 0.65, 0.36))                     # touches the expanded cashier box
    far = Det("phone", 0.9, (0.05, 0.05, 0.1, 0.1))                      # in the visitor's hands
    landline = Det("phone", 0.15, (0.6, 0.5, 0.64, 0.52))                # next to the cashier, but a weak score
    dets = [[landline], [near], [near], [far, landline], [near], [landline]]
    ev = object_events(timeline(dets), None, None, people, [cashier, visitor])
    assert kinds(ev) == [(1.0, "phone_near_cashier"), (4.0, "phone_near_cashier")]
    assert object_events(timeline(dets), None, None, replace(people, cashier_id=None), [cashier]) == []


def test_keys_near_safe():
    keys = Det("keys", 0.2, (0.8, 0.3, 0.82, 0.32))
    elsewhere = Det("keys", 0.5, (0.1, 0.9, 0.12, 0.92))
    ev = object_events(timeline([[], [elsewhere], [keys], [keys], []]), None, SAFE, NO_PEOPLE, [])
    assert kinds(ev) == [(2.0, "keys_near_safe")]


def test_best_detection_window_and_zone_filter():
    tl = timeline([[CASH], [], [Det("cash", 0.8, (0.0, 0.0, 0.1, 0.1))], [], []], step=0.5)
    assert tl.best("cash", 1.0).score == 0.8                              # 0.0..2.0 s window
    assert tl.best("cash", 1.0, within=DESK) == CASH
    assert tl.best("cash", 2.0, tol=0.5) is None
    assert tl.best("phone", 0.0) is None


def frames(n: int, fps: float = 5.0) -> VideoFrames:
    return VideoFrames([round(i / fps, 3) for i in range(n)], [np.zeros((8, 8, 3), np.uint8)] * n, (n - 1) / fps)


def test_key_times_grid_plus_snapped_extras_capped():
    vf = frames(51)                                                         # 10 s at 5 fps
    ts = key_times(vf, 1.0, [3.4, 3.6, 4.2, 5.62, 7.0, 7.21, -1.0, 99.0])
    assert ts == sorted([float(i) for i in range(11)] + [3.4, 5.6])        # extras < 0.4 s from a chosen frame dropped
    assert len(key_times(vf, 1.0, [i * 0.1 for i in range(100)], cap=13)) == 13
    assert len(key_times(vf, 5.0, [], cap=20)) <= 20


def test_detect_timeline_stops_at_deadline():
    vf = frames(51)
    det = FakeObjectDetector(lambda im: [CASH])
    tl, truncated = detect_timeline(vf, det, [float(i) for i in range(10)], batch=4, deadline=time.monotonic() - 1)
    assert truncated and tl.times == [0.0, 1.0, 2.0, 3.0] and det.calls == 4
    tl, truncated = detect_timeline(vf, det, [float(i) for i in range(10)], batch=4)
    assert not truncated and len(tl.dets) == 10 and tl.dets[0] == [CASH]



def owl_with_raw_outputs(pred: list[tuple[str, float, tuple]], n_boxes: int = 16) -> OwlDetector:
    """OwlDetector with the processor and model replaced: pred = [(query, score, (cx, cy, w, h) in the padded square)]."""
    import math
    from types import SimpleNamespace

    import torch
    from torchvision.ops import batched_nms

    queries = list(QUERIES)
    det = OwlDetector.__new__(OwlDetector)
    det.labels = sorted(set(QUERIES.values()))
    det.torch, det.nms, det.device, det.threshold, det.nms_iou, det.dtype = torch, batched_nms, "cpu", 0.13, 0.5, torch.float32
    det.ids = det.mask = torch.ones((len(queries), 2), dtype=torch.long)
    det.group = torch.tensor([det.labels.index(g) for g in QUERIES.values()])
    logits, boxes = torch.full((1, n_boxes, len(queries)), -20.0), torch.full((1, n_boxes, 4), 0.05)
    for i, (q, s, b) in enumerate(pred):
        logits[0, i, queries.index(q)], boxes[0, i] = math.log(s / (1 - s)), torch.tensor(b)
    det.processor = SimpleNamespace(image_processor=lambda images, return_tensors, device: {
        "pixel_values": torch.zeros(len(images), 3, 8, 8)})
    det.model = lambda input_ids, attention_mask, pixel_values: SimpleNamespace(
        logits=logits.repeat(len(pixel_values), 1, 1), pred_boxes=boxes.repeat(len(pixel_values), 1, 1))
    return det


def test_owl_boxes_map_back_from_the_padded_square():
    pred = [("stack of banknotes", 0.14, (0.25, 0.5, 0.1, 0.1)),             # weakest true hit (cash in hands)
            ("laptop", 0.7, (0.8, 0.5, 0.2, 0.2))]                            # right half of the square
    pred += [("mobile phone", 0.3 - 0.01 * i, (0.03 + 0.07 * i, 0.9, 0.04, 0.04)) for i in range(7)]
    det = owl_with_raw_outputs(pred)
    portrait = det.detect([np.zeros((848, 464, 3), np.uint8)])[0]           # 464 x 848: padded on the right
    k = 848 / 464
    assert next(d for d in portrait if d.label == "cash").box == pytest.approx((0.2 * k, 0.45, 0.3 * k, 0.55), abs=1e-4)
    assert "laptop" not in {d.label for d in portrait}                        # centre in the padding: dropped
    assert [d.score for d in portrait if d.label == "phone"] == [0.3, 0.29, 0.28, 0.27, 0.26]   # per-label cap
    landscape = det.detect([np.zeros((720, 1280, 3), np.uint8)])[0]          # padded at the bottom
    k = 1280 / 720
    assert next(d for d in landscape if d.label == "cash").box == pytest.approx((0.2, 0.45 * k, 0.3, 0.55 * k), abs=1e-4)
    assert next(d for d in landscape if d.label == "laptop").box == pytest.approx((0.7, 0.4 * k, 0.9, 1.0), abs=1e-4)  # clipped
    assert not any(d.label == "phone" for d in landscape)                     # y = 0.9 * 1.78: below the image

NORM = {"act": "Правила № 231", "point": "п. 20"}
FACTS = {f.id: f for f in (Fact("F_cash_desk", "desk", "q"), Fact("F_bag_desk", "desk", "q"),
                           Fact("F_phone", "full", "q"), Fact("F_safe_open", "safe", "q"))}
ZONES = Zones(desk=DESK, safe=SAFE)


def rule(rid: str, facts: list[str]) -> Rule:
    return Rule(rid, "Тип", ["duration"], facts, {}, NORM, "В", "В {t} с (кадр {te} с).")


def test_object_region_from_detection_near_evidence_time():
    tl = timeline([[], [], [], [], [CASH, Det("phone", 0.2, (0.5, 0.0, 0.6, 0.1))], [Det("phone", 0.3, (0.0, 0.0, 0.1, 0.1))]])
    r01 = rule("R01", ["F_cash_desk", "F_bag_desk"])
    region = object_region(r01, Candidate("R01", 3.0, 4.5, "F_cash_desk", 0.9), tl, list(DESK))
    assert region == [0.292, 0.593, 0.407, 0.708]                         # cash box, x1.15
    assert object_region(r01, Candidate("R01", 3.0, 2.0, "F_cash_desk", 0.9), tl, list(DESK)) is None  # > 1 s away
    r11 = rule("R11", ["F_phone"])                                         # full-frame rule: any phone box counts
    assert object_region(r11, Candidate("R11", 5.0, 5.0, "F_phone", 0.9), tl, None) == [0.0, 0.0, 0.107, 0.107]
    assert object_region(r11, Candidate("R11", 3.5, 3.5, "F_phone", 0.9), tl, None) is None   # 0.2 < FREE_MIN_SCORE
    assert object_region(rule("R99", ["F_cash_desk"]), Candidate("R99", 4.0, 4.0, None, 0.9), tl, None) is None
    assert object_region(r01, Candidate("R01", 4.0, 4.0, "F_cash_desk", 0.9), None, list(DESK)) is None


def test_object_region_never_leaves_the_rule_zone_and_follows_the_fact():
    tl = timeline([[Det("cash", 0.9, (0.0, 0.0, 0.05, 0.05)), Det("bag", 0.3, (0.5, 0.6, 0.7, 0.75))]])
    r01 = rule("R01", ["F_cash_desk", "F_bag_desk"])
    assert object_region(r01, Candidate("R01", 0.0, 0.0, "F_cash_desk", 0.9), tl, list(DESK)) is None
    assert object_region(r01, Candidate("R01", 0.0, 0.0, "F_bag_desk", 0.9), tl, list(DESK)) == [0.485, 0.589, 0.715, 0.761]


def test_build_violations_prefers_object_box_and_falls_back_to_zone():
    vf = VideoFrames([i * 0.2 for i in range(100)], [np.full((60, 40, 3), 100, np.uint8)] * 100, 20.0)
    rules = {"R01": rule("R01", ["F_cash_desk"]), "R02": rule("R02", ["F_safe_open"])}
    acc = [(Candidate("R01", 4.0, 4.0, "F_cash_desk", 0.9), 0.9), (Candidate("R02", 12.0, 12.0, "F_safe_open", 0.9), 0.8)]
    tl = ObjectTimeline([4.0, 12.0], [[CASH], [Det("safe", 0.5, (0.0, 0.9, 0.1, 1.0))]])   # that "safe" is elsewhere
    by_rule = {v.rule_id: v for v in build_violations(acc, rules, vf, zones=ZONES, facts_by_id=FACTS, objects=tl)}
    assert by_rule["R01"].region == [0.292, 0.593, 0.407, 0.708]
    assert by_rule["R02"].region == [0.7, 0.1, 0.95, 0.4]                  # zone fallback
    old = {v.rule_id: v for v in build_violations(acc, rules, vf, zones=ZONES, facts_by_id=FACTS)}
    assert old["R01"].region == list(DESK)


# --- run.py integration --------------------------------------------------------------------------------------------
W, H = 96, 160
RULES = load_rules(Path("app/rules.yaml"))
FACTS_ALL = load_facts(Path("app/facts.yaml"))


def painter_phone(i):
    """test_run.painter + a green 'phone' next to the cashier (above the desk zone) at 3.0-3.97 s."""
    img = painter(i)
    if 90 <= i < 120:
        img[45:55, 62:70] = (0, 220, 0)
    return img


def owl_like(im):
    out = []
    if (im[110:130, 10:40].min(axis=2) > 200).mean() > 0.5:
        out.append(Det("cash", 0.5, (10 / W, 110 / H, 40 / W, 130 / H)))
    if im[50, 66, 1] > 150 and im[50, 66, 0] < 80:
        out.append(Det("phone", 0.3, (62 / W, 45 / H, 70 / W, 55 / H)))
    return out


@pytest.fixture
def captured_extra(monkeypatch):
    seen: list[list[float]] = []
    real = run_mod.probe_times

    def spy(duration, fps, extra, cap):
        seen.append(list(extra))
        return real(duration, fps, extra, cap)

    monkeypatch.setattr(run_mod, "probe_times", spy)
    return seen


def test_analyze_with_fake_object_detector(tmp_video, captured_extra):
    path = tmp_video(450, 30, painter_phone, size=(W, H))                 # 15 s
    det = FakeObjectDetector(owl_like)
    verdict, violations, debug = analyze(path, Models(cash_vlm(), BlockDetector(), det), SETTINGS, RULES, FACTS_ALL)
    events = {(e["kind"], round(e["t"])) for e in debug["obj_events"]}
    assert ("cash_on_desk_appeared", 6) in events and ("phone_near_cashier", 3) in events
    assert det.calls == debug["obj_frames"] >= 16 and "t_objects" in debug and debug["objects_truncated"] is False
    assert {2.6, 2.8, 3.2, 3.4} <= set(captured_extra[0])                 # Qwen probes around the phone event
    r01 = next(v for v in violations if v.rule_id == "R01")
    assert verdict == "violation" and r01.region == [0.081, 0.678, 0.44, 0.822]   # cash box, not the whole desk


def test_analyze_with_detector_off_has_no_object_keys(tmp_video, captured_extra):
    path = tmp_video(450, 30, painter_phone, size=(W, H))
    verdict, violations, debug = analyze(path, Models(cash_vlm(), BlockDetector()), SETTINGS, RULES, FACTS_ALL)
    assert not {"obj_events", "obj_frames", "t_objects", "objects_error", "objects_skipped"} & set(debug)
    assert 2.6 not in captured_extra[0]
    assert verdict == "violation" and next(v for v in violations if v.rule_id == "R01").region == [0.0, 0.6, 1.0, 0.9]


def test_object_detector_failure_does_not_fail_analyze(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))

    def boom(im):
        raise RuntimeError("CUDA out of memory")

    verdict, violations, debug = analyze(path, Models(cash_vlm(), BlockDetector(), FakeObjectDetector(boom)),
                                         SETTINGS, RULES, FACTS_ALL)
    assert "CUDA out of memory" in debug["objects_error"] and "obj_events" not in debug
    assert verdict == "violation" and any(v.rule_id == "R01" for v in violations)


def test_object_stage_skipped_late_in_budget(tmp_video):
    path = tmp_video(150, 30, painter, size=(W, H))
    det = FakeObjectDetector(owl_like)
    _, _, debug = analyze(path, Models(cash_vlm(), BlockDetector(), det), SETTINGS, RULES, FACTS_ALL, budget_s=0.0)
    assert debug["objects_skipped"] is True and det.calls == 0


def test_build_models_object_detector_switch(monkeypatch):
    monkeypatch.setattr(run_mod, "TorchvisionPersonDetector", lambda *a: object())
    s = replace(Settings(), vlm_backend="fake")
    assert build_models(s).objects is None                                 # default "off"

    def broken(*a):
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(run_mod, "OwlDetector", broken)
    assert build_models(replace(s, obj_detector="owlv2")).objects is None   # logged, service still starts
    monkeypatch.setattr(run_mod, "OwlDetector", lambda *a: ("owl", a))
    assert build_models(replace(s, obj_detector="owlv2", device="cuda")).objects == (
        "owl", ("google/owlv2-base-patch16-ensemble", "cuda", 0.13))


def test_object_settings_from_env(monkeypatch):
    assert (Settings().obj_detector, Settings().obj_threshold, Settings().obj_fps) == ("off", 0.13, 1.0)
    for k, v in {"OBJ_DETECTOR": "owlv2", "OBJ_THRESHOLD": "0.2", "OBJ_FPS": "2", "OBJ_MODEL": "m"}.items():
        monkeypatch.setenv(k, v)
    s = load_settings()
    assert (s.obj_detector, s.obj_threshold, s.obj_fps, s.obj_model) == ("owlv2", 0.2, 2.0, "m")
