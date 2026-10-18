from dataclasses import replace
from pathlib import Path

from app.pipeline.people import Event, PeopleTimeline
from app.pipeline.rules import RuleContext, evaluate, load_rules
from app.pipeline.timeline import Interval

RULES = {r.id: r for r in load_rules(Path("app/rules.yaml"), include_disabled=True)}


def ctx(events, states, others=lambda t: 0, series=None, dur=30.0):
    times = [round(i * 0.2, 3) for i in range(int(dur * 5))]
    p = PeopleTimeline(times, 1, [False] * len(times), [others(t) for t in times], events)
    return RuleContext(p, states, series or {}, [(t, 120.0, 50.0) for t in times], dur)


def iv(a, b):
    return Interval(a, b, a, 0.9, 5)


def test_r13_counting_after_client_left():
    c = evaluate([RULES["R13"]], ctx([Event(11.0, "exited", 2)], {"F_counting": [iv(12.5, 16.0)]}))
    assert [x.t for x in c] == [12.5]


def test_r13_not_when_counting_before_exit():
    assert evaluate([RULES["R13"]], ctx([Event(11.0, "exited", 2)], {"F_counting": [iv(5.0, 9.0)]})) == []


CLIENT_IN = [Event(2.0, "entered", 2)]                     # the client's entry is in the clip (R14 need_entry)


def test_r14_handover_without_id():
    c = evaluate([RULES["R14"]], ctx(CLIENT_IN, {"F_handover": [iv(10.0, 11.0)], "F_id_card": []}, others=lambda t: 1))
    assert [x.t for x in c] == [10.0]
    assert evaluate([RULES["R14"]], ctx(CLIENT_IN, {"F_handover": [iv(10.0, 11.0)], "F_id_card": [iv(4.0, 8.0)]}, others=lambda t: 1)) == []


def test_r14_needs_the_clients_entry_in_the_clip():
    """X-3, X-19, X-39 (clean): the client is at the desk from 0 s, hands money over at 7-8 s, no ID in the clip; the ID
    may have been shown before the recording started."""
    s = {"F_handover": [iv(7.2, 8.0)]}
    assert evaluate([RULES["R14"]], ctx([], s, others=lambda t: 1)) == []
    # X-3: a passer-by entered at 4.0 and left at 6.8, before the handover: still the client from 0 s
    assert evaluate([RULES["R14"]], ctx([Event(4.0, "entered", 3), Event(6.8, "exited", 3)], s, others=lambda t: 1)) == []
    # X-2 / X-38: the client walked in during the clip and is still there
    assert [x.t for x in evaluate([RULES["R14"]], ctx([Event(1.4, "entered", 3)], s, others=lambda t: 1))] == [7.2]
    off = replace(RULES["R14"], params=dict(RULES["R14"].params, need_entry=False))
    assert [x.t for x in evaluate([off], ctx([], s, others=lambda t: 1))] == [7.2]


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
    assert evaluate([RULES["R14"]], ctx(CLIENT_IN, {"F_handover": [iv(10.0, 11.0)], "F_id_card": [iv(10.4, 12.0)]}, others=present)) == []
    assert evaluate([RULES["R14"]], ctx(CLIENT_IN, {"F_handover": [iv(30.0, 31.0)], "F_id_card": [iv(2.0, 6.0)]}, others=present, dur=40.0)) == []


def test_r14_not_without_a_client():
    assert evaluate([RULES["R14"]], ctx(CLIENT_IN, {"F_handover": [iv(10.0, 11.0)]}, others=lambda t: 0)) == []


def test_r15_not_for_a_long_visit():
    events = [Event(3.0, "entered", 5), Event(20.0, "exited", 5)]
    assert evaluate([RULES["R15"]], ctx(events, {"F_bag_desk": [iv(6.5, 30.0)]})) == []


def test_tier2_candidates_are_never_strong():
    c = evaluate([RULES["R14"]], ctx(CLIENT_IN, {"F_handover": [iv(10.0, 11.0)]}, others=lambda t: 1))
    assert c and not any(x.strong for x in c)


R20_STATES = {"F_cash_from_client": [iv(5.0, 8.0)], "F_counting": [iv(9.0, 18.0)], "F_cash_to_client": [iv(20.0, 22.0)]}
PRESENT = lambda t: 1


def r20(states, events=(), others=PRESENT, **kw):
    return [x.t for x in evaluate([RULES["R20"]], ctx(list(events), states, others=others, **kw))]


def test_r20_part_of_counted_deposit_goes_back_to_client():
    c = evaluate([RULES["R20"]], ctx([], R20_STATES, others=PRESENT))
    assert [x.t for x in c] == [20.0]
    assert c[0].fact == "F_cash_to_client" and not c[0].strong


def test_r20_not_for_a_withdrawal_without_deposit():
    assert r20({"F_counting": [iv(9.0, 18.0)], "F_cash_to_client": [iv(20.0, 22.0)]}) == []


def test_r20_not_when_client_already_left():
    assert r20(R20_STATES, others=lambda t: 1 if t < 19.0 else 0) == []


def test_r20_not_when_money_goes_out_long_after_counting():
    assert r20(dict(R20_STATES, F_cash_to_client=[iv(58.0, 59.0)]), dur=60.0) == []


def test_r20_not_when_counting_starts_long_after_deposit():
    states = {"F_cash_from_client": [iv(2.0, 4.0)], "F_counting": [iv(36.0, 45.0)], "F_cash_to_client": [iv(48.0, 50.0)]}
    assert r20(states, dur=60.0) == []


def test_r20_not_when_cashier_left_between_deposit_and_return():
    assert r20(R20_STATES, events=[Event(19.0, "cashier_left", 1)]) == []


def test_r20_not_for_a_mirror_answer_during_the_deposit_handover():
    states = {"F_cash_from_client": [iv(5.0, 8.0)], "F_counting": [iv(6.0, 18.0)], "F_cash_to_client": [iv(7.0, 8.0)]}
    assert r20(states) == []


def test_r20_not_for_a_return_before_counting_got_going():
    states = {"F_cash_from_client": [iv(5.0, 8.0)], "F_counting": [iv(12.0, 20.0)], "F_cash_to_client": [iv(10.0, 11.0)]}
    assert r20(states) == []
    assert r20(dict(states, F_cash_to_client=[iv(13.0, 14.0)])) == []        # < min_count_s of counting


def test_r20_not_for_the_next_client_withdrawal():
    """Client A deposits and leaves; client B arrives and is paid out (X-5 has an exit at 60.6 and an entry at 62.0)."""
    states = {"F_cash_from_client": [iv(2.0, 4.0)], "F_counting": [iv(5.0, 10.0), iv(16.0, 22.0)],
              "F_cash_to_client": [iv(24.0, 25.0)]}
    events = [Event(11.0, "exited", 2), Event(14.0, "entered", 3)]
    assert r20(states, events=events, others=lambda t: 0 if 11.0 <= t < 14.0 else 1) == []
    assert r20(states, events=events[1:]) == []                              # B joins before A has left


def test_r20_counting_may_start_just_before_the_deposit_interval():
    states = {"F_cash_from_client": [iv(5.0, 8.0)], "F_counting": [iv(4.5, 18.0)], "F_cash_to_client": [iv(20.0, 22.0)]}
    assert r20(states) == [20.0]
    assert r20(dict(states, F_counting=[iv(2.0, 18.0)])) == []               # counting began well before: not this deposit


def test_r20_falls_through_to_a_later_counting_of_the_same_client():
    states = {"F_cash_from_client": [iv(5.0, 8.0)], "F_counting": [iv(9.0, 12.0), iv(14.0, 20.0)],
              "F_cash_to_client": [iv(38.0, 39.0)]}
    assert r20(states, dur=60.0) == [38.0]


def test_r20_every_return_after_count_is_a_candidate():
    """An early to-client blip (rejected by S7) must not hide the real return (X-1: 22.4 s)."""
    states = dict(R20_STATES, F_cash_to_client=[iv(12.0, 13.0), iv(22.4, 24.0)])
    assert r20(states) == [12.0, 22.4]


def test_r20_is_never_strong_even_with_a_confident_series():
    series = {"F_cash_to_client": [(float(t), 0.95) for t in range(15, 30)]}
    c = evaluate([RULES["R20"]], ctx([], R20_STATES, others=PRESENT, series=series))
    assert [x.t for x in c] == [20.0] and c[0].score == 0.95 and not c[0].strong
    weak_k = replace(RULES["R20"], params=dict(RULES["R20"].params, strong_k=3))
    assert evaluate([weak_k], ctx([], R20_STATES, others=PRESENT, series=series))[0].strong   # strong_k: 99 is what blocks it


def test_r14_s7_needs_the_handover_itself():
    """X-5@54.8 and X-35@7.6 (clean) passed S7 on the two "no ID seen" answers alone while the handover question said
    no; every R14 S7 question is now a veto (measured Qwen3-VL-8B answers)."""
    from app.pipeline.verify import decide
    r = RULES["R14"]
    assert all(v.get("required") for v in r.verify)
    assert decide(r, [0.004, 0.0, 0.002])[0] is False             # X-5@54.8: no money passed in the clip
    assert decide(r, [0.438, 0.0, 0.023])[0] is False             # X-35@7.6
    assert decide(r, [0.9, 0.05, 0.1])[0] is True                 # cash handed over, no ID in either clip
