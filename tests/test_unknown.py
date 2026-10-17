import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.people import Event, PeopleTimeline
from app.pipeline.rules import Rule
from app.pipeline.unknown import scan_unknown
from app.pipeline.vlm import FakeVLM

VF = VideoFrames([i * 0.2 for i in range(100)], [np.zeros((40, 30, 3), np.uint8)] * 100, 20.0)
PEOPLE = PeopleTimeline(VF.times, 1, [False] * 100, [0] * 100, [Event(5.0, "entered", 2)])
R = Rule("R05", "Кассир покинул место", ["cashier_left_with_client"], [], {}, {"act": "a", "point": "p"}, "В", "{t}")


def test_description_matched_to_rule():
    vlm = FakeVLM(answer=lambda imgs, q: 0.95 if "R05" in q else 0.1, reply=lambda imgs, p: "The cashier walks away while a client waits.")
    out = scan_unknown(VF, PEOPLE, [R], vlm, already=set())
    assert [(c.rule_id, c.t) for c, _ in out] == [("R05", 5.0)]


def test_already_found_rules_skipped():
    vlm = FakeVLM(answer=lambda imgs, q: 0.95, reply=lambda imgs, p: "x")
    assert scan_unknown(VF, PEOPLE, [R], vlm, already={"R05"}) == []


def test_best_hit_per_rule_across_events():
    import time  # noqa: F401
    people = PeopleTimeline(VF.times, 1, [False] * 100, [0] * 100, [Event(5.0, "entered", 2), Event(15.0, "exited", 2)])
    probs = iter([0.9, 0.95])
    vlm = FakeVLM(answer=lambda imgs, q: next(probs), reply=lambda imgs, p: "x")
    out = scan_unknown(VF, people, [R], vlm, already=set())
    assert len(out) == 1 and out[0][0].t == 15.0 and out[0][1] == 0.95


def test_deadline_in_the_past_makes_no_calls():
    import time
    vlm = FakeVLM(answer=lambda imgs, q: 0.95, reply=lambda imgs, p: "x")
    assert scan_unknown(VF, PEOPLE, [R], vlm, already=set(), deadline=time.monotonic() - 1) == []
    assert vlm.calls == 0


def test_none_description_does_not_crash():
    vlm = FakeVLM(answer=lambda imgs, q: 0.1, reply=lambda imgs, p: None)
    assert scan_unknown(VF, PEOPLE, [R], vlm, already=set()) == []
