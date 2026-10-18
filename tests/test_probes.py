from pathlib import Path

import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.people import PeopleTimeline
from app.pipeline.probes import Fact, load_facts, probe_times, run_probes
from app.pipeline.scene import Zones
from app.pipeline.vlm import FakeVLM


def test_facts_yaml_loads():
    facts = load_facts(Path("app/facts.yaml"))
    ids = {f.id for f in facts}
    assert {"F_cash_desk", "F_safe_open", "F_screen_on", "F_handover"} <= ids
    # every fact feeds a rule (or R14's F_id_card); the door is read from pixels (R08), a VLM door probe was dead weight
    rules = Path("app/rules.yaml").read_text(encoding="utf-8")
    assert all(f in rules or f == "F_id_card" for f in ids), sorted(f for f in ids if f not in rules)
    assert all(f.zone in {"desk", "safe", "door", "laptop", "full", "cashier"} for f in facts)
    assert next(f for f in facts if f.id == "F_phone").zone == "cashier"     # X-5: the client's phone is not hers


def test_probe_times_base_plus_extra_and_cap():
    assert probe_times(3.0, 1.0, [1.4, 1.4, 9.0], cap=100) == [0.0, 1.0, 1.4, 2.0, 3.0]
    ts = probe_times(10.0, 1.0, [i / 10 for i in range(100)], cap=15)
    assert len(ts) == 15 and ts[0] == 0.0 and 10.0 in ts


def test_probe_times_long_video_subsamples_base():
    assert len(probe_times(500.0, 1.0, [], cap=100)) <= 100


def test_run_probes_series_follow_frames():
    imgs = [np.full((100, 60, 3), 255 if i >= 5 else 0, np.uint8) for i in range(10)]
    vf = VideoFrames([float(i) for i in range(10)], imgs, 9.0)
    facts = [Fact("F_cash_desk", "desk", "banknotes?"), Fact("F_door_open", "door", "door?")]
    vlm = FakeVLM(answer=lambda ims, q: 0.9 if ("banknotes" in q and ims[0].mean() > 100) else 0.1)
    s = run_probes(vf, Zones(desk=(0, 0.5, 1, 1)), vlm, facts, [0.0, 4.0, 5.0, 9.0])
    assert [p for _, p in s["F_cash_desk"]] == [0.1, 0.1, 0.9, 0.9]
    assert [p for _, p in s["F_door_open"]] == [0.1] * 4


def _cashier_people(times, boxes, cashier=1):
    return PeopleTimeline(times, cashier, [b is None for b in boxes], [0] * len(times), [], None, boxes)


def test_cashier_zone_crops_to_the_cashier_box():
    """F_phone on the cashier's person box, tight and upscaled: the client's phone across the desk is not in it (X-5)."""
    img = np.zeros((200, 100, 3), np.uint8)
    img[150:200, :] = 255                                          # the client's side of the frame is white
    vf = VideoFrames([0.0, 1.0, 2.0], [img] * 3, 2.0)
    fact = [Fact("F_phone", "cashier", "phone?")]
    shapes = []

    def answer(ims, q):
        shapes.append(ims[0].shape[:2])
        return 0.9 if ims[0].mean() > 0 else 0.1                   # "yes" whenever the white client side is visible

    people = _cashier_people([0.0, 1.0, 2.0], [(0.2, 0.1, 0.8, 0.5), None, (0.2, 0.1, 0.8, 0.5)])
    s = run_probes(vf, Zones(), FakeVLM(answer=answer), fact, [0.0, 1.0, 2.0], people)
    assert s["F_phone"] == [(0.0, 0.1), (1.0, 0.0), (2.0, 0.1)]   # out of view at 1.0: P = 0 without a question
    assert shapes == [(597, 448), (597, 448)]                       # the 80 x 60 px box, no margin, short side 448
    no_cashier = _cashier_people([0.0, 1.0, 2.0], [None] * 3, cashier=None)
    assert run_probes(vf, Zones(), FakeVLM(answer=answer), fact, [0.0], no_cashier)["F_phone"] == [(0.0, 0.9)]
    assert run_probes(vf, Zones(), FakeVLM(answer=answer), fact, [0.0])["F_phone"] == [(0.0, 0.9)]   # full frame


def test_cashier_box_at_the_frame_edge_is_out_of_view():
    """A box squeezed against the bottom edge crops to zero rows: no question, P = 0 (not a ZeroDivisionError)."""
    img = np.zeros((200, 100, 3), np.uint8)
    vf = VideoFrames([0.0], [img], 1.0)
    people = _cashier_people([0.0], [(0.2, 1.0, 0.8, 1.0)])
    vlm = FakeVLM(answer=lambda ims, q: 0.9)
    assert run_probes(vf, Zones(), vlm, [Fact("F_phone", "cashier", "phone?")], [0.0], people)["F_phone"] == [(0.0, 0.0)]
    assert vlm.calls == 0


def test_event_anchored_fact_is_asked_only_near_its_events():
    """F_screen_on, F_key, F_docs_desk, F_stamps_desk feed on_cashier_left rules only: no sample farther than NEAR_S
    from a departure; without S2 people they are asked everywhere."""
    from app.pipeline.people import Event
    from app.pipeline.probes import NEAR_S
    img = np.zeros((40, 40, 3), np.uint8)
    ts = [float(t) for t in range(0, 41, 5)]
    vf = VideoFrames(ts, [img] * len(ts), 40.0)
    facts = [Fact("F_screen_on", "full", "screen?", ("cashier_left",)), Fact("F_cash_desk", "full", "cash?")]
    people = PeopleTimeline(ts, 1, [False] * len(ts), [0] * len(ts), [Event(20.0, "cashier_left", 1)])
    vlm = FakeVLM(answer=lambda ims, q: 0.7)
    s = run_probes(vf, Zones(), vlm, facts, ts, people)
    assert [t for t, _ in s["F_screen_on"]] == [t for t in ts if abs(t - 20.0) <= NEAR_S] == [10.0, 15.0, 20.0, 25.0, 30.0]
    assert [t for t, _ in s["F_cash_desk"]] == ts
    assert vlm.calls == 5 + len(ts)
    nobody = PeopleTimeline(ts, 1, [False] * len(ts), [0] * len(ts), [])
    assert run_probes(vf, Zones(), vlm, facts, ts, nobody)["F_screen_on"] == []
    assert [t for t, _ in run_probes(vf, Zones(), vlm, facts, ts)["F_screen_on"]] == ts


def test_facts_yaml_near_lists_only_on_cashier_left_facts():
    facts = {f.id: f for f in load_facts(Path("app/facts.yaml"))}
    assert {k for k, f in facts.items() if f.near} == {"F_docs_desk", "F_stamps_desk", "F_key", "F_screen_on"}
    assert all(f.near == ("cashier_left",) for f in facts.values() if f.near)
    from app.pipeline.rules import load_rules                      # a rule of another family would get a sparse series
    for r in load_rules(Path("app/rules.yaml"), include_disabled=True):
        if any(facts[f].near for f in r.facts if f in facts):
            assert r.families == ["on_cashier_left"], r.id
