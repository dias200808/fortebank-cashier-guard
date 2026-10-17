from pathlib import Path

from app.pipeline.people import Event, PeopleTimeline
from app.pipeline.rules import RuleContext, evaluate, load_rules
from app.pipeline.timeline import Interval

T = [round(i * 0.2, 3) for i in range(150)]
RULES = {r.id: r for r in load_rules(Path("app/rules.yaml"))}


def ctx(events, states, others=lambda t: 0):
    p = PeopleTimeline(T, 1, [False] * len(T), [others(t) for t in T], events)
    return RuleContext(p, states, {}, [(t, 120.0, 50.0) for t in T], 30.0)


def iv(a, b):
    return Interval(a, b, a, 0.9, 5)


def test_r13_counting_after_client_left():
    c = evaluate([RULES["R13"]], ctx([Event(11.0, "exited", 2)], {"F_counting": [iv(12.5, 16.0)]}))
    assert [x.t for x in c] == [12.5]


def test_r13_not_when_counting_before_exit():
    assert evaluate([RULES["R13"]], ctx([Event(11.0, "exited", 2)], {"F_counting": [iv(5.0, 9.0)]})) == []


def test_r14_handover_without_id():
    c = evaluate([RULES["R14"]], ctx([], {"F_handover": [iv(10.0, 11.0)], "F_id_card": []}, others=lambda t: 1))
    assert [x.t for x in c] == [10.0]
    assert evaluate([RULES["R14"]], ctx([], {"F_handover": [iv(10.0, 11.0)], "F_id_card": [iv(4.0, 8.0)]}, others=lambda t: 1)) == []


def test_r15_collector_leaves_quickly():
    events = [Event(3.0, "entered", 5), Event(9.0, "exited", 5)]
    c = evaluate([RULES["R15"]], ctx(events, {"F_bag_desk": [iv(6.5, 30.0)]}))
    assert [x.t for x in c] == [6.5]


def test_r13_not_after_collector_leaves_with_bag_on_desk():
    events = [Event(3.0, "entered", 7), Event(11.0, "exited", 7)]
    c = evaluate([RULES["R13"]], ctx(events, {"F_counting": [iv(13.0, 20.0)], "F_bag_desk": [iv(6.0, 30.0)]}))
    assert c == []


def test_r13_not_when_someone_is_still_present():
    c = evaluate([RULES["R13"]], ctx([Event(11.0, "exited", 2)], {"F_counting": [iv(12.5, 16.0)]}, others=lambda t: 1))
    assert c == []


def test_r14_not_when_id_shown_at_handover_or_long_before():
    present = lambda t: 1
    assert evaluate([RULES["R14"]], ctx([], {"F_handover": [iv(10.0, 11.0)], "F_id_card": [iv(10.4, 12.0)]}, others=present)) == []
    assert evaluate([RULES["R14"]], ctx([], {"F_handover": [iv(30.0, 31.0)], "F_id_card": [iv(2.0, 6.0)]}, others=present)) == []


def test_r14_not_without_a_client():
    assert evaluate([RULES["R14"]], ctx([], {"F_handover": [iv(10.0, 11.0)]}, others=lambda t: 0)) == []


def test_r15_not_for_a_long_visit():
    events = [Event(3.0, "entered", 5), Event(20.0, "exited", 5)]
    assert evaluate([RULES["R15"]], ctx(events, {"F_bag_desk": [iv(6.5, 30.0)]})) == []


def test_tier2_candidates_are_never_strong():
    c = evaluate([RULES["R14"]], ctx([], {"F_handover": [iv(10.0, 11.0)]}, others=lambda t: 1))
    assert c and not any(x.strong for x in c)
