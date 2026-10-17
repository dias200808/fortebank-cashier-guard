import re
from pathlib import Path

from app.pipeline.people import Event, PeopleTimeline
from app.pipeline.rules import FAMILIES, RuleContext, evaluate, load_rules
from app.pipeline.timeline import Interval, intervals

T = [round(i * 0.2, 3) for i in range(150)]        # 30 s
RULES = {r.id: r for r in load_rules(Path("app/rules.yaml"))}


def people(events, away_from=None, others=None):
    away = [away_from is not None and t >= away_from for t in T]
    oth = [others(t) if others else 0 for t in T]
    return PeopleTimeline(T, 1, away, oth, events)


def ctx(p, states=None, quality=None, series=None):
    return RuleContext(p, states or {}, series or {}, quality or [(t, 120.0, 50.0) for t in T], 30.0)


def iv(a, b, peak=0.9, strong=5):
    return Interval(a, b, (a + b) / 2, peak, strong)


def test_catalog_is_complete():
    for r in RULES.values():
        assert r.type and r.norm["act"] and r.norm["point"] and r.strength in {"В", "С", "Н"}
        assert all(f in FAMILIES for f in r.families), r.id
        assert "{t}" in r.template


def test_r01_cash_on_desk_when_cashier_leaves():
    series = {"F_cash_desk": [(float(t), 0.95) for t in range(8, 30)]}
    c = evaluate([RULES["R01"]], ctx(people([Event(9.0, "cashier_left", 1)], away_from=9.0), {"F_cash_desk": [iv(8.6, 30)]},
                                     series=series))
    assert len(c) == 1 and c[0].rule_id == "R01" and c[0].t == 9.0 and c[0].strong


def test_r01_silent_when_desk_empty():
    assert evaluate([RULES["R01"]], ctx(people([Event(9.0, "cashier_left", 1)], away_from=9.0), {"F_cash_desk": []})) == []


def test_r02_open_safe_while_cashier_absent():
    c = evaluate([RULES["R02"]], ctx(people([], away_from=10.0), {"F_safe_open": [iv(2.0, 20.0)]}))
    assert [x.t for x in c] == [10.0]


def test_r02_open_safe_with_cashier_present_is_fine():
    assert evaluate([RULES["R02"]], ctx(people([]), {"F_safe_open": [iv(2.0, 20.0)]})) == []


def test_r05_left_while_client_present():
    p = people([Event(9.0, "cashier_left", 1)], away_from=9.0, others=lambda t: 1)
    assert [x.rule_id for x in evaluate([RULES["R05"]], ctx(p))] == ["R05"]


def test_r06_entry_with_cash_from_previous_operation():
    p = people([Event(12.0, "entered", 2)], others=lambda t: 1 if t >= 12.0 else 0)
    c = evaluate([RULES["R06"]], ctx(p, {"F_cash_desk": [iv(5.0, 20.0)]}))
    assert [x.t for x in c] == [12.0]


def test_r06_not_when_previous_client_still_present():
    p = people([Event(12.0, "entered", 3)], others=lambda t: 1)
    assert evaluate([RULES["R06"]], ctx(p, {"F_cash_desk": [iv(5.0, 20.0)]})) == []


def _door(ctx_, spans, step=0.2, unknown=()):
    ctx_.door = [(round(i * step, 1), -1.0 if any(a <= i * step < b for a, b in unknown)
                  else (0.4 if any(a <= i * step < b for a, b in spans) else 0.01)) for i in range(int(ctx_.duration / step) + 1)]
    return ctx_


def test_r08_door_left_open_from_pixels():
    # opened at 4 s and never closed: timecode = onset + 2 s, no VLM verification needed
    assert [x.t for x in evaluate([RULES["R08"]], _door(ctx(people([])), [(4.0, 31.0)]))] == [6.0]
    # opened for a passage and closed again
    assert evaluate([RULES["R08"]], _door(ctx(people([])), [(4.0, 9.0)])) == []
    # lights switched off / door hidden by a person: unknown samples neither extend nor break the run
    assert [x.t for x in evaluate([RULES["R08"]], _door(ctx(people([])), [(4.0, 31.0)], unknown=[(8.0, 12.0)]))] == [6.0]
    assert evaluate([RULES["R08"]], _door(ctx(people([])), [], unknown=[(20.0, 31.0)])) == []
    # no door zone -> no series -> nothing (the full frame is not a door)
    assert evaluate([RULES["R08"]], ctx(people([]), {"F_door_open": [iv(0.0, 30.0)]})) == []


def test_r08_open_until_video_end_counts_from_end_min():
    assert [x.t for x in evaluate([RULES["R08"]], _door(ctx(people([])), [(25.0, 31.0)]))] == [27.0]
    assert evaluate([RULES["R08"]], _door(ctx(people([])), [(28.0, 31.0)])) == []


def test_r09_second_visitor_enters():
    p = people([Event(17.0, "entered", 3)], others=lambda t: 2 if t >= 17.0 else 1)
    assert [x.t for x in evaluate([RULES["R09"]], ctx(p))] == [17.0]


def test_r10_presence_needs_min_duration():
    assert [x.t for x in evaluate([RULES["R10"]], ctx(people([]), {"F_items": [iv(15.0, 25.0)]}))] == [15.0]
    assert evaluate([RULES["R10"]], ctx(people([]), {"F_items": [iv(15.0, 15.6)]})) == []


def test_quality_rule_dark_video():
    q = [(t, 5.0 if t >= 10.0 else 120.0, 50.0) for t in T]
    assert [x.t for x in evaluate([RULES["R18"]], ctx(people([]), quality=q))] == [10.0]


def test_quotes_are_verbatim_from_research():
    research = Path("docs/normative_research.md").read_text(encoding="utf-8").replace("**", "")
    for r in RULES.values():
        for piece in re.split(r"[…;]", r.norm.get("quote", "")):
            piece = piece.strip(" ,.«»")
            if len(piece) >= 15:
                assert piece in research, (r.id, piece)


def test_r09_not_for_two_people_arriving_together():
    p = people([Event(10.0, "entered", 2), Event(11.5, "entered", 3)],
               others=lambda t: 0 if t < 10.0 else (1 if t < 11.5 else 2))
    assert evaluate([RULES["R09"]], ctx(p)) == []


def test_r05_not_when_visitor_just_arrived():
    p = people([Event(9.0, "cashier_left", 1)], away_from=9.0, others=lambda t: 1 if t >= 8.0 else 0)
    assert evaluate([RULES["R05"]], ctx(p)) == []


def test_r02_not_for_short_away_runs_while_safe_open():
    away = [10.0 <= t < 11.0 or 20.0 <= t < 21.0 for t in T]
    p = PeopleTimeline(T, 1, away, [0] * len(T), [])
    assert evaluate([RULES["R02"]], ctx(p, {"F_safe_open": [iv(5.0, 29.0)]})) == []


def test_r02_needs_a_tracked_cashier():
    p = PeopleTimeline(T, None, [True] * len(T), [0] * len(T), [])
    assert evaluate([RULES["R02"]], ctx(p, {"F_safe_open": [iv(5.0, 29.0)]})) == []


def test_r02_departure_not_duplicated():
    c = evaluate([RULES["R02"]], ctx(people([Event(10.0, "cashier_left", 1)], away_from=10.0), {"F_safe_open": [iv(2.0, 20.0)]}))
    assert len(c) == 1 and c[0].t == 10.0


def test_r01_not_when_cash_removed_before_leaving():
    p = people([Event(9.4, "cashier_left", 1)], away_from=9.4)
    assert evaluate([RULES["R01"]], ctx(p, {"F_cash_desk": [iv(5.0, 8.0)]})) == []


def test_r04_strength_and_evidence_come_from_after_the_departure():
    series = [(float(t), 0.95 if t <= 10 else 0.05) for t in range(30)]
    p = people([Event(10.3, "cashier_left", 1)], away_from=10.3)
    c = evaluate([RULES["R04"]], ctx(p, {"F_screen_on": intervals(series)}, series={"F_screen_on": series}))
    assert len(c) == 1 and c[0].strong is False and c[0].evidence_t >= 10.3


def test_strong_counts_base_grid_samples_only():
    series = [(10.0, 0.1), (10.2, 0.95), (10.4, 0.95), (10.6, 0.95), (11.0, 0.1), (12.0, 0.1), (13.0, 0.1)]
    p = people([Event(10.0, "cashier_left", 1)], away_from=10.0)
    c = evaluate([RULES["R01"]], ctx(p, {"F_cash_desk": [iv(8.6, 30)]}, series={"F_cash_desk": series}))
    assert len(c) == 1 and c[0].strong is False


def test_r04_cites_point_53():
    assert RULES["R04"].norm["point"] == "п. 53"
    assert RULES["R04"].norm["quote"] == "определяются банком, организацией"


def dark_from(t0):
    return [(t, 5.0 if t >= t0 else 120.0, 50.0) for t in T]


def test_r18_not_when_light_goes_off_after_cashier_left():
    assert evaluate([RULES["R18"]], ctx(people([], away_from=9.0), quality=dark_from(10.0))) == []


def test_r18_fires_when_nobody_is_ever_tracked():
    p = PeopleTimeline(T, None, [True] * len(T), [0] * len(T), [])
    assert [x.t for x in evaluate([RULES["R18"]], ctx(p, quality=dark_from(0.0)))] == [0.0]


def test_r18_fires_when_lens_covered_with_cashier_present():
    assert [x.t for x in evaluate([RULES["R18"]], ctx(people([], away_from=9.8), quality=dark_from(10.0)))] == [10.0]


def test_r18_fires_when_covered_from_start_and_cashier_seen_later():
    p = PeopleTimeline(T, 1, [t < 9.0 for t in T], [0] * len(T), [])
    q = [(t, 5.0 if t < 8.0 else 120.0, 50.0) for t in T]
    assert [x.t for x in evaluate([RULES["R18"]], ctx(p, quality=q))] == [0.0]


def _series(a, b, step=0.5):
    return [(round(i * step, 2), 0.9 if a <= i * step <= b else 0.05) for i in range(int(30 / step) + 1)]


def test_r02_safe_left_open_while_cashier_back_at_desk():
    # X-20: safe opened at 2 s, cashier stands at it until 3.5 s, then sits at the desk; the safe stays open
    c = ctx(people([]), {"F_safe_open": [iv(2.0, 29.0)], "F_safe_attended": [iv(0.0, 3.5)]},
            series={"F_safe_open": _series(2.0, 29.0)})
    assert [x.t for x in evaluate([RULES["R02"]], c)] == [4.0]
    # normal use: somebody stands at the safe the whole time it is open
    c = ctx(people([]), {"F_safe_open": [iv(10.0, 14.0)], "F_safe_attended": [iv(9.0, 15.0)]},
            series={"F_safe_open": _series(10.0, 14.0)})
    assert evaluate([RULES["R02"]], c) == []


def test_r12_cup_from_first_second_needs_drinking():
    assert evaluate([RULES["R12"]], ctx(people([]), {"F_cup": [iv(0.0, 20.0)]})) == []
    assert [x.t for x in evaluate([RULES["R12"]], ctx(people([]), {"F_cup": [iv(0.0, 20.0)], "F_drinking": [iv(8.0, 9.0)]}))] == [0.0]
    assert [x.t for x in evaluate([RULES["R12"]], ctx(people([]), {"F_cup": [iv(11.0, 20.0)]}))] == [11.0]


def test_r14_id_shown_after_the_money_counts():
    p = people([], others=lambda t: 1)
    assert evaluate([RULES["R14"]], ctx(p, {"F_handover": [iv(12.0, 13.0)], "F_id_card": [iv(16.0, 17.5)]})) == []
    assert [x.t for x in evaluate([RULES["R14"]], ctx(p, {"F_handover": [iv(12.0, 13.0)]}))] == [12.0]
