from pathlib import Path

from app.pipeline.people import Event, PeopleTimeline
from app.pipeline.probes import load_facts
from app.pipeline.rules import FAMILIES, RuleContext, evaluate, load_rules
from app.pipeline.timeline import Interval

RULES = {r.id: r for r in load_rules(Path("app/rules.yaml"))}
FACT_IDS = {f.id for f in load_facts(Path("app/facts.yaml"))}


def ctx(states, events=(), away=lambda t: False, others=lambda t: 0, dur=30.0, series=None, cashier=1):
    times = [round(i * 0.2, 3) for i in range(int(dur * 5) + 1)]
    p = PeopleTimeline(times, cashier, [away(t) for t in times], [others(t) for t in times], list(events))
    return RuleContext(p, states, series or {}, [(t, 120.0, 50.0) for t in times], dur)


def iv(a, b):
    return Interval(a, b, a, 0.9, 5)


def run(rule_ids, c):
    return [(x.rule_id, x.t) for x in evaluate([RULES[r] for r in rule_ids], c)]


def test_every_rule_has_summary_and_signals_for_the_catalog():
    for r in RULES.values():
        assert r.summary.strip() and r.signals.strip(), r.id
        assert r.summary != r.type and r.signals != r.summary, r.id


def test_every_rule_uses_registered_families_and_probed_facts():
    for r in RULES.values():
        assert all(f in FAMILIES for f in r.families), r.id
        attend = r.params.get("attend_fact", [])
        used = set(r.facts) | set(r.params.get("count_facts", [])) | set([attend] if isinstance(attend, str) else attend)
        assert used <= FACT_IDS, (r.id, used - FACT_IDS)


def test_new_rules_are_in_the_catalog():
    assert {"R07", "R16", "R21", "R22"} <= set(RULES)
    assert RULES["R21"].strength == RULES["R22"].strength == RULES["R16"].strength == "Н"
    assert RULES["R07"].strength == "С"
    assert "appeared_while_away" in RULES["R01"].families
    for rid in ("R07", "R16", "R21", "R22"):
        assert RULES[rid].params["strong_k"] == 99 and len(RULES[rid].verify) >= 2   # only through S7
    for rid in ("R16", "R22"):
        assert RULES[rid].params.get("s8") is False              # S7 cannot see "not counted" / "safe opened before"
    assert [v["q"] for v in RULES["R22"].verify if v.get("required")] == ["Is the door of the safe open?"]


# --- R01: cash or a bag put on the desk after the cashier left (appeared_while_away) ---

LEFT = [Event(9.0, "cashier_left", 1)]


def visitor(t):
    return 1 if 12.0 <= t <= 16.0 else 0                         # somebody comes in and puts the item down


def test_r01_bag_put_on_desk_while_cashier_away():
    c = ctx({"F_bag_desk": [iv(14.0, 25.0)]}, LEFT, away=lambda t: t >= 9.0, others=visitor)
    assert run(["R01"], c) == [("R01", 14.0)]


def test_r01_not_when_nobody_was_in_the_room_to_put_it_there():
    """Proper departure (X-12/25/27/29): the desk is empty and the room too; a later desk interval is probe noise."""
    assert run(["R01"], ctx({"F_cash_desk": [iv(14.0, 25.0)]}, LEFT, away=lambda t: t >= 9.0)) == []


def test_r01_item_already_on_desk_at_departure_is_not_duplicated():
    c = ctx({"F_bag_desk": [iv(10.0, 25.0)], "F_cash_desk": [iv(8.6, 30.0)]}, LEFT, away=lambda t: t >= 9.0,
            others=lambda t: 1)
    assert run(["R01"], c) == [("R01", 9.0)]                     # on_cashier_left owns items within tol


def test_r01_not_when_cashier_came_back_before_the_item_appeared():
    c = ctx({"F_bag_desk": [iv(14.0, 25.0)]}, LEFT, away=lambda t: 9.0 <= t < 12.0, others=visitor)
    assert run(["R01"], c) == []


def test_r01_not_without_a_departure_event():
    # standing up (away flags) is not a departure: cashier_left needs >= 2 s out of view or leaving for good
    assert run(["R01"], ctx({"F_bag_desk": [iv(14.0, 25.0)]}, [], away=lambda t: t >= 9.0, others=visitor)) == []


def test_r01_short_blip_mid_video_does_not_count():
    assert run(["R01"], ctx({"F_cash_desk": [iv(14.0, 16.0)]}, LEFT, away=lambda t: t >= 9.0, others=visitor)) == []


def test_r01_item_appearing_at_the_end_of_a_short_clip():
    c = ctx({"F_bag_desk": [iv(6.5, 8.8)]}, [Event(3.0, "cashier_left", 1)], away=lambda t: t >= 3.0, dur=9.0,
            others=lambda t: 1 if t >= 5.0 else 0)
    assert run(["R01"], c) == [("R01", 6.5)]


# --- R07: the next client's money goes onto the previous client's cash (deposit_onto_leftover) ---

X4_EVENTS = [Event(4.8, "entered", 2), Event(53.6, "exited", 2), Event(58.8, "entered", 3)]


def x4_others(t):
    return 0 if 53.6 <= t < 58.8 or t < 4.8 else 1


def test_r07_deposit_onto_cash_left_by_previous_client():
    states = {"F_cash_desk": [iv(57.0, 68.0)], "F_handover": [iv(60.5, 61.5), iv(65.5, 66.8)]}
    c = evaluate([RULES["R07"]], ctx(states, X4_EVENTS, others=x4_others, dur=73.0))
    assert [x.t for x in c] == [66.8]                            # latest handover: the money, not the ID
    assert c[0].fact == "F_cash_desk" and not c[0].strong


def test_r07_cash_from_client_counts_as_the_deposit():
    states = {"F_cash_desk": [iv(57.0, 68.0)], "F_cash_from_client": [iv(63.0, 64.0)]}
    assert run(["R07"], ctx(states, X4_EVENTS, others=x4_others, dur=73.0)) == [("R07", 64.0)]


def test_r07_not_when_previous_client_still_present():
    states = {"F_cash_desk": [iv(57.0, 68.0)], "F_handover": [iv(65.5, 66.8)]}
    assert run(["R07"], ctx(states, [Event(58.8, "entered", 3)], others=lambda t: 1, dur=73.0)) == []


def test_r07_not_when_desk_was_cleared_or_cash_came_with_the_new_client():
    hand = {"F_handover": [iv(65.5, 66.8)]}
    for cash in ([iv(50.0, 57.5)], [iv(59.0, 68.0)]):
        assert run(["R07"], ctx(dict(hand, F_cash_desk=cash), X4_EVENTS, others=x4_others, dur=73.0)) == []


def test_r07_not_for_a_handover_right_at_entry_or_after_the_cash_was_put_away():
    early = {"F_cash_desk": [iv(57.0, 68.0)], "F_handover": [iv(59.5, 60.0)]}
    late = {"F_cash_desk": [iv(57.0, 62.0)], "F_handover": [iv(64.0, 65.0)]}
    assert run(["R07"], ctx(early, X4_EVENTS, others=x4_others, dur=73.0)) == []
    assert run(["R07"], ctx(late, X4_EVENTS, others=x4_others, dur=73.0)) == []


def test_r07_x5_two_clients_with_a_cleared_desk_is_silent():
    """X-5: exit 60.6, entry 62.0 s, the desk is clean in between."""
    events = [Event(60.6, "exited", 2), Event(62.0, "entered", 3)]
    states = {"F_cash_desk": [iv(40.0, 59.0)], "F_handover": [iv(66.0, 67.0)]}
    assert run(["R07"], ctx(states, events, others=lambda t: 0 if 60.6 <= t < 62.0 else 1, dur=72.0)) == []


# --- R16: cash from the desk into the safe without counting (desk_to_safe_without_count) ---

X13 = {"F_cash_desk": [iv(23.0, 26.0), iv(40.0, 43.0)], "F_safe_attended": [iv(36.0, 40.0), iv(45.0, 57.0)]}


def test_r16_cash_briefly_on_desk_then_into_safe():
    c = evaluate([RULES["R16"]], ctx(X13, dur=65.0, others=lambda t: 1 if 39.6 <= t <= 54.0 else 0))  # reflection
    assert [x.t for x in c] == [46.0] and c[0].fact == "F_safe_attended" and not c[0].strong


def test_r16_not_after_counting():
    assert run(["R16"], ctx(dict(X13, F_counting=[iv(30.0, 35.0)]), dur=65.0)) == []


def test_r16_not_for_money_received_from_a_client():
    assert run(["R16"], ctx(dict(X13, F_handover=[iv(20.0, 21.0)]), dur=65.0)) == []
    assert run(["R16"], ctx(dict(X13, F_cash_from_client=[iv(38.0, 39.0)]), dur=65.0)) == []


def test_r16_not_when_cash_lay_on_the_desk_long_enough_to_count():
    """X-14: manual count, the cash lies on the desk 46-54 and 104-114 s before going to the safe."""
    states = {"F_cash_desk": [iv(46.0, 51.0), iv(52.0, 54.0), iv(104.0, 108.0), iv(109.0, 114.0)],
              "F_safe_attended": [iv(116.0, 122.0)]}
    assert run(["R16"], ctx(states, dur=142.0)) == []


def test_r16_not_for_a_withdrawal_counted_or_paid_out_after_the_safe_visit():
    """Safe -> desk -> back to the safe to close it -> count and pay out: the client operation is the п. 42 exception."""
    assert run(["R16"], ctx(dict(X13, F_cash_to_client=[iv(58.0, 59.0)]), dur=70.0)) == []
    assert run(["R16"], ctx(dict(X13, F_counting=[iv(50.0, 55.0)]), dur=70.0)) == []


def test_r16_hand_count_silences_short_handling_does_not():
    """F_counting misses hand counts (X-14): banknotes held >= 8 s are a count; carrying them 5 s (X-13) is not."""
    assert run(["R16"], ctx(dict(X13, F_cash_hands=[iv(28.0, 38.0)]), dur=65.0)) == []
    assert run(["R16"], ctx(dict(X13, F_cash_hands=[iv(40.0, 45.0)]), dur=65.0)) == [("R16", 46.0)]


def test_r16_not_without_a_safe_visit_right_after():
    assert run(["R16"], ctx({"F_cash_desk": [iv(40.0, 43.0)], "F_safe_attended": [iv(50.0, 57.0)]}, dur=65.0)) == []
    assert run(["R16"], ctx({"F_cash_desk": [iv(40.0, 43.0)]}, dur=65.0)) == []


# --- R21: the laptop screen shown to a visitor (presence of F_screen_shown) ---

def test_r21_screen_shown_on_a_short_clip():
    assert run(["R21"], ctx({"F_screen_shown": [iv(6.0, 9.5)]}, dur=10.0, others=lambda t: 1)) == [("R21", 6.0)]
    assert run(["R21"], ctx({"F_screen_shown": [iv(6.0, 6.6)]}, dur=10.0, others=lambda t: 1)) == []


def test_r21_needs_a_visitor_in_the_room():
    assert run(["R21"], ctx({"F_screen_shown": [iv(6.0, 9.5)]}, dur=10.0)) == []


# --- R22: cash counted alone while the safe stands open (count_alone_open_safe) ---

X20 = {"F_safe_open": [iv(2.0, 11.6)], "F_safe_attended": [iv(0.0, 4.0)], "F_cash_hands": [iv(6.0, 11.4)]}


def test_r22_counting_alone_with_the_safe_left_open():
    c = evaluate([RULES["R22"]], ctx(X20, dur=11.6))
    assert [x.t for x in c] == [6.0] and c[0].fact == "F_cash_hands" and not c[0].strong


def test_r22_reported_in_addition_to_r02():
    series = {"F_safe_open": [(float(t), 0.95 if t >= 2 else 0.05) for t in range(12)]}
    assert {r for r, _ in run(["R02", "R22"], ctx(X20, dur=11.6, series=series))} == {"R02", "R22"}


def test_r22_counting_machine_signal_also_counts():
    states = dict(X20, F_cash_hands=[], F_counting=[iv(7.0, 11.0)])
    assert run(["R22"], ctx(states, dur=11.6)) == [("R22", 7.0)]


def test_r22_x21_safe_closed_before_counting_is_normal():
    """X-21 (clean): the same solo count, but the safe was closed at 7 s before counting began."""
    states = {"F_safe_open": [iv(1.0, 7.0)], "F_safe_attended": [iv(0.0, 7.5)], "F_cash_hands": [iv(5.0, 12.0)]}
    assert run(["R22"], ctx(states, dur=19.6)) == []
    assert run(["R22"], ctx(dict(states, F_cash_hands=[iv(9.0, 12.0)]), dur=19.6)) == []


def test_r22_not_when_safe_opened_only_after_counting_started():
    assert run(["R22"], ctx(dict(X20, F_safe_open=[iv(7.0, 11.6)]), dur=11.6)) == []


def test_r22_not_while_someone_stands_at_or_reaches_into_the_safe():
    """X-14@118: the cashier holds banknotes while putting them into the open safe."""
    assert run(["R22"], ctx(dict(X20, F_safe_attended=[iv(0.0, 11.6)]), dur=11.6)) == []
    assert run(["R22"], ctx(dict(X20, F_safe_touched=[iv(5.0, 11.6)]), dur=11.6)) == []


def test_r22_not_with_a_client_or_another_person_in_the_room():
    assert run(["R22"], ctx(X20, dur=11.6, others=lambda t: 1)) == []
    # a short reflection blip does not count as a visitor
    assert run(["R22"], ctx(X20, dur=11.6, others=lambda t: 1 if 8.0 <= t <= 8.4 else 0)) == [("R22", 6.0)]


def test_r22_client_operation_is_the_p42_exception():
    for fact in ("F_cash_from_client", "F_cash_to_client", "F_handover"):
        assert run(["R22"], ctx(dict(X20, **{fact: [iv(1.0, 2.0)]}), dur=11.6)) == [], fact
    payout = dict(X20, F_cash_to_client=[iv(30.0, 31.0)])                   # paid out to a client 20 s later
    assert run(["R22"], ctx(payout, dur=40.0)) == []


def test_r22_not_for_a_collection_bag():
    assert run(["R22"], ctx(dict(X20, F_bag_desk=[iv(1.0, 3.0)]), dur=11.6)) == []


def test_r22_not_when_cashier_away_or_untracked():
    assert run(["R22"], ctx(X20, dur=11.6, away=lambda t: t >= 5.0)) == []
    assert run(["R22"], ctx(X20, dur=11.6, cashier=None)) == []


def test_r22_needs_min_duration():
    assert run(["R22"], ctx(dict(X20, F_cash_hands=[iv(6.0, 7.4)]), dur=11.6)) == []


def test_r22_s7_rejects_a_closed_safe_whatever_the_counting_answers():
    """X-21 through S7: counting alone at the desk, nobody watching, but the safe is closed -> rejected."""
    from app.pipeline.verify import decide
    r = RULES["R22"]
    ans = {"Is the door of the safe open?": 0.1}
    probs = [ans.get(v["q"], 0.9 if v.get("expect", "yes") == "yes" else 0.1) for v in r.verify]
    assert decide(r, probs)[0] is False
    ans = {"Is the door of the safe open?": 0.9, "Is the cashier sitting at the desk holding a stack of banknotes?": 0.3}
    probs = [ans.get(v["q"], 0.9 if v.get("expect", "yes") == "yes" else 0.1) for v in r.verify]
    assert decide(r, probs)[0] is True                           # X-20: open safe + 2 of the other 3
