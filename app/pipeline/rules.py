"""S6: rule catalog + evaluation of CV events x VLM states -> violation candidates."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from app.pipeline.people import PeopleTimeline
from app.pipeline.timeline import Interval, Series, active


@dataclass
class Rule:
    id: str
    type: str
    families: list[str]
    facts: list[str]
    params: dict
    norm: dict
    strength: str
    template: str
    verify: list[dict] = field(default_factory=list)
    enabled: bool = True
    summary: str = ""          # what the violation is, for the catalog on the website and in README
    signals: str = ""          # which signals the rule combines, same audience


@dataclass
class Candidate:
    rule_id: str
    t: float
    evidence_t: float
    fact: str | None
    score: float
    strong: bool = False


@dataclass
class RuleContext:
    people: PeopleTimeline
    states: dict[str, list[Interval]]
    series: dict[str, Series]
    quality: list[tuple[float, float, float]]     # (t, brightness, sharpness)
    duration: float
    probe_fps: float = 1.0                         # base probe grid; dense extras between grid points
    door: list[tuple[float, float]] = field(default_factory=list)  # door panel change vs first frame, -1 unknown


def load_rules(path: Path, include_disabled: bool = False) -> list[Rule]:
    rules = []
    for d in yaml.safe_load(Path(path).read_text(encoding="utf-8")):
        fam = d.pop("family")
        d["families"] = fam if isinstance(fam, list) else [fam]
        rules.append(Rule(**d))
    return [r for r in rules if r.enabled or include_disabled]


STRONG_P = 0.8


def _cand(rule: Rule, fact: str, iv: Interval, t: float) -> Candidate:
    return Candidate(rule.id, t, iv.peak_t, fact, iv.peak_p, iv.strong >= rule.params.get("strong_k", 3))


def _on_grid(t: float, fps: float) -> bool:
    """t is a base-grid probe time (probe_times rounds grid points i/fps to 2 decimals)."""
    return abs(t - round(round(t * fps) / fps, 2)) < 1e-6


def _post(rule: Rule, ctx: RuleContext, fact: str, t0: float, t1: float) -> tuple[float, float, bool]:
    """(evidence_t, score, strong) from the probe series inside [t0, t1]; strong = >= strong_k consecutive
    base-grid samples with p >= STRONG_P (dense extras are excluded so k means k seconds at 1 fps)."""
    pts = [(t, p) for t, p in ctx.series.get(fact, []) if t0 <= t <= t1]
    if not pts:
        return t0, 0.0, False
    evidence_t, score = max(pts, key=lambda x: x[1])
    run = best = 0
    for t, p in pts:
        if not _on_grid(t, ctx.probe_fps):
            continue
        run = run + 1 if p >= STRONG_P else 0
        best = max(best, run)
    return evidence_t, score, best >= rule.params.get("strong_k", 3)


def _post_cand(rule: Rule, ctx: RuleContext, fact: str, t: float, t0: float, t1: float) -> Candidate:
    evidence_t, score, strong = _post(rule, ctx, fact, t0, t1)
    return Candidate(rule.id, t, evidence_t, fact, score, strong)


def on_cashier_left(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    out = []
    for ev in ctx.people.events:
        if ev.kind != "cashier_left":
            continue
        for f in rule.facts:
            tol, after_tol = rule.params.get("tol", 1.5), rule.params.get("after_tol", 0.5)
            iv = next((i for i in ctx.states.get(f, []) if i.start - tol <= ev.t <= i.end + after_tol), None)
            if iv:
                out.append(_post_cand(rule, ctx, f, ev.t, ev.t, ev.t + rule.params.get("post_s", 5.0)))
                break
    return out


def appeared_while_away(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    """Cash or a bag put on the desk after the cashier left, while she has not come back (a collector or a client leaves
    it in the empty room). Intervals starting within tol of a departure belong to on_cashier_left (same tol), so the two
    families never report the same item. A plain stand-up is no departure: cashier_left needs >= 2 s out of view or
    leaving for good. An interval that runs to the end of a short clip counts from end_min_s. Somebody must have been in
    the room to put it there: with nobody but the absent cashier the "new" item is probe noise on the empty desk left
    after a proper departure (X-12/25/27/29)."""
    p = rule.params
    tol, min_s, end_min = p.get("tol", 1.5), p.get("min_s", 3.0), p.get("end_min_s", 2.0)
    left = [e.t for e in ctx.people.events if e.kind == "cashier_left"]
    out = []
    for f in rule.facts:
        for iv in ctx.states.get(f, []):
            ev = max((t for t in left if t < iv.start - tol), default=None)
            need = min_s if iv.end < ctx.duration - 1.0 else min(min_s, end_min)
            if ev is None or iv.length < need:
                continue
            ts = [t for t in ctx.people.times if ev <= t <= iv.start + need]
            if all(ctx.people.away_at(t) for t in ts) and any(ctx.people.others_at(t) > 0 for t in ts):
                out.append(_post_cand(rule, ctx, f, iv.start, iv.start, iv.end))   # not back yet, a visitor was in
    return out


def unattended_state(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    if ctx.people.cashier_id is None:
        return []
    out, min_s = [], rule.params.get("min_s", 3.0)
    for f in rule.facts:
        for iv in ctx.states.get(f, []):
            run: list[float] = []
            for t in [x for x in ctx.people.times if iv.start <= x <= iv.end] + [None]:
                if t is not None and ctx.people.away_at(t):
                    run.append(t)
                    continue
                if run and run[-1] - run[0] >= min_s:
                    out.append(_post_cand(rule, ctx, f, run[0], run[0], iv.end))
                    break
                run = []
    return out


def cashier_left_with_client(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    stay, served = rule.params.get("client_stays_s", 2.0), rule.params.get("served_s", 5.0)
    return [Candidate(rule.id, ev.t, ev.t + 1.0, None, 0.7, False)
            for ev in ctx.people.events
            if ev.kind == "cashier_left" and ctx.people.others_at(ev.t) > 0 and ctx.people.others_at(ev.t + stay) > 0
            and ctx.people.others_at(ev.t - served) > 0]


def on_entry(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    out, before = [], rule.params.get("before_s", 1.5)
    for ev in ctx.people.events:
        if ev.kind != "entered" or ctx.people.others_at(ev.t - before) > 0:
            continue                                   # previous visitor still here -> same operation
        for f in rule.facts:
            iv = active(ctx.states.get(f, []), ev.t - before / 2, before / 2)
            if iv and iv.start <= ev.t - 0.5:
                out.append(_post_cand(rule, ctx, f, ev.t, ev.t - 1.0, ev.t + 2.0))
                break
    return out


def entry_during_service(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    stay, served = rule.params.get("min_stay_s", 1.0), rule.params.get("served_s", 5.0)
    return [Candidate(rule.id, ev.t, ev.t + 1.0, None, 0.6, False)
            for ev in ctx.people.events
            if ev.kind == "entered" and ctx.people.others_at(ev.t - 1.0) >= 1 and ctx.people.others_at(ev.t + stay) >= 2
            and ctx.people.others_at(ev.t - served) >= 1]


def duration(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    min_s = rule.params.get("min_s", 8.0)
    return [_post_cand(rule, ctx, f, iv.start, iv.start, iv.end)
            for f in rule.facts for iv in ctx.states.get(f, []) if iv.length >= min_s]


def presence(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    """An object visible from the first second may be furniture the model mistakes for it (teal boxes taken for cups
    in X-25): then the start_needs fact (e.g. someone drinking) must be seen somewhere in the video. need_visitor: the
    state concerns a visitor (the screen shown to one), so S2 must see somebody besides the cashier during it."""
    min_s, need, visitor = rule.params.get("min_s", 2.0), rule.params.get("start_needs"), rule.params.get("need_visitor")
    return [_post_cand(rule, ctx, f, iv.start, iv.start, iv.end)
            for f in rule.facts for iv in ctx.states.get(f, [])
            if iv.length >= min_s and (not need or iv.start > 1.0 or ctx.states.get(need))
            and (not visitor or any(ctx.people.others_at(t) > 0 for t in ctx.people.times if iv.start <= t <= iv.end))]


def state_left_unattended(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    """The state holds while nobody stands at it for min_s: the cashier opened the safe, went back to the desk and left
    it open (X-20). Unlike unattended_state the cashier may still be in the room; normal use of the safe keeps one of
    the attend_fact facts (standing next to it / touching or reaching into it, even from the chair as in X-39) on."""
    names = rule.params.get("attend_fact", [])
    att = [a for n in ([names] if isinstance(names, str) else names) for a in ctx.states.get(n, [])]
    min_s = rule.params.get("min_s", 3.0)
    out = []
    for f in rule.facts:
        for iv in ctx.states.get(f, []):
            run: list[float] = []
            for t in [x for x, _ in ctx.series.get(f, []) if iv.start <= x <= iv.end] + [None]:
                if t is not None and not any(a.start <= t <= a.end for a in att):
                    run.append(t)
                    continue
                if run and run[-1] - run[0] >= min_s:
                    out.append(_post_cand(rule, ctx, f, run[0], run[0], iv.end))
                    break
                run = []
    return out


def quality(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    p = rule.params
    bad = [b < p.get("dark", 25.0) or s < p.get("flat", 10.0) for _, b, s in ctx.quality]
    out, start = [], None
    for (t, _, _), is_bad in zip(ctx.quality + [(ctx.duration + 1, 255.0, 255.0)], bad + [False]):
        if is_bad and start is None:
            start = t
        elif not is_bad and start is not None:
            # light switched off after the cashier had already left -> end of shift, not a covered camera
            left_before = (ctx.people.cashier_id is not None and start - 1.0 >= ctx.people.times[0]
                           and ctx.people.away_at(start - 1.0))
            if t - start >= p.get("min_s", 2.0) and not left_before:
                out.append(Candidate(rule.id, start, start, None, 0.9, True))
            start = None
    return out


def count_after_client_left(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    out = []
    for ev in ctx.people.events:
        if ev.kind != "exited":
            continue
        if ctx.people.others_at(ev.t + rule.params.get("after_s", 1.0)) > 0:
            continue                                   # client still present -> not counted after leaving
        if active(ctx.states.get("F_bag_desk", []), ev.t, 0.5):
            continue                                   # collector bag on desk -> normal bag handling
        for f in rule.facts:
            for iv in ctx.states.get(f, []):
                if ev.t + rule.params.get("after_s", 1.0) <= iv.start <= ev.t + 10.0:
                    out.append(_cand(rule, f, iv, iv.start))
    return out


def handover_without_id(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    look, ahead = rule.params.get("lookback_s", 20.0), rule.params.get("lookahead_s", 3.0)
    ids = ctx.states.get("F_id_card", [])
    out = []
    for f in rule.facts:
        for iv in ctx.states.get(f, []):
            if ctx.people.others_at(iv.start) == 0:
                continue                               # no client at the handover -> nothing to identify
            if not any(x.end >= iv.start - look and x.start <= iv.end + ahead for x in ids):  # ID may follow the money
                out.append(_cand(rule, f, iv, iv.start))
    return out


def collector_quick_leave(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    max_stay = rule.params.get("max_stay_s", 10.0)
    entries = {e.track_id: e.t for e in ctx.people.events if e.kind == "entered"}
    out = []
    for ev in ctx.people.events:
        if ev.kind != "exited" or ev.track_id not in entries or ev.t - entries[ev.track_id] > max_stay:
            continue
        for f in rule.facts:
            for iv in ctx.states.get(f, []):
                if entries[ev.track_id] <= iv.start <= ev.t and iv.end >= ev.t:
                    out.append(_cand(rule, f, iv, iv.start))
    return out


def return_after_count(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    """Deposit -> counting -> part of the money goes back across the desk to the same, still-present client (X-1).
    Withdrawals (count, then pay out) have no preceding deposit and are not matched. The return must start after
    the deposit handover (the mirror to-client question can fire on the client's own handover) and after some
    counting; any visitor entry/exit in between means another client (two operations in a row). Every matching
    return is a candidate, so an S7 rejection of an early blip cannot hide the real return."""
    p = rule.params
    lead, count_within = p.get("count_lead_s", 2.0), p.get("count_within_s", 30.0)
    gap, min_count, return_within = p.get("after_deposit_s", 1.0), p.get("min_count_s", 3.0), p.get("return_within_s", 20.0)
    left = [e.t for e in ctx.people.events if e.kind == "cashier_left"]
    visits = [e.t for e in ctx.people.events if e.kind in ("entered", "exited")]
    counts, returns = ctx.states.get("F_counting", []), ctx.states.get("F_cash_to_client", [])
    hits: set[float] = set()
    for d in ctx.states.get("F_cash_from_client", []):
        for c in counts:
            if not (d.start - lead <= c.start <= d.end + count_within and c.end > d.end):
                continue                               # counting must belong to this deposit and outlast the handover
            for r in returns:
                if (max(d.end + gap, c.start + min_count) <= r.start <= c.end + return_within
                        and ctx.people.others_at(r.start) > 0                  # client still at the desk
                        and not any(d.start <= t <= r.start for t in left)     # cashier stayed
                        and not any(d.start < t <= r.start for t in visits)):  # same client throughout
                    hits.add(r.start)
    return [_post_cand(rule, ctx, "F_cash_to_client", t, t, t + 3.0) for t in sorted(hits)]


def deposit_onto_leftover(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    """X-4@68: the previous client's cash is still on the desk when the next client enters (the on_entry / R06
    condition) and, before it is put away, the new client hands money across the desk. The latest handover in the
    window is taken: an early one is usually the ID being shown. One candidate per entry, always checked by S7."""
    p = rule.params
    before, min_after = p.get("before_s", 1.5), p.get("min_after_s", 2.0)
    within, tol = p.get("within_s", 60.0), p.get("tol", 1.5)
    handovers = [h for n in ("F_cash_from_client", "F_handover") for h in ctx.states.get(n, [])]
    out = []
    for e in ctx.people.events:
        if e.kind != "entered" or ctx.people.others_at(e.t - before) > 0:
            continue                                   # previous visitor still here -> same operation
        left = next((c for c in ctx.states.get("F_cash_desk", []) if c.start <= e.t - 0.5 and c.end >= e.t), None)
        if left is None:
            continue
        hs = [h for h in handovers if e.t + min_after <= h.start <= min(left.end + tol, e.t + within)
              and ctx.people.others_at(h.start) > 0]
        if hs:
            h = max(hs, key=lambda x: x.start)
            t = round(min(h.end, left.end), 1)
            out.append(_post_cand(rule, ctx, "F_cash_desk", t, t - 3.0, left.end + 0.5))
    return out


def desk_to_safe_without_count(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    """X-13@47: banknotes lie on the desk only a few seconds, nobody counts them, and right after that the cashier is at
    the safe. A long manual count (X-14: on the desk 46-54 and 104-114 s) fails the total-time limit; money that came
    from a client across the desk is the п. 42 exception. People counts are not used on purpose: a reflection track on
    X-13 (39.6-54.0 s) flickers others_at = 1 and would kill the labelled case. A withdrawal runs the other way (safe ->
    desk -> back to the safe to close it -> count and pay out), so counting or a client cash flow within after_s of the
    safe visit also silences the rule; F_counting misses hand counts (X-14), so banknotes held >= hands_min_s count too."""
    p = rule.params
    max_desk, max_total, look = p.get("max_desk_s", 6.0), p.get("max_total_s", 12.0), p.get("lookback_s", 60.0)
    gap, client_s, lag = p.get("gap_s", 5.0), p.get("client_s", 30.0), p.get("lag_s", 1.0)
    after, hands_min = p.get("after_s", 20.0), p.get("hands_min_s", 8.0)
    cash, counts = ctx.states.get("F_cash_desk", []), ctx.states.get("F_counting", [])
    hands = [x for x in ctx.states.get("F_cash_hands", []) if x.length >= hands_min]
    clients = [x for n in ("F_cash_from_client", "F_cash_to_client", "F_handover") for x in ctx.states.get(n, [])]
    out = []
    for c in cash:
        total = sum(x.length for x in cash if x.end >= c.start - look and x.start <= c.end)
        if c.length > max_desk or total > max_total:
            continue                                   # on the desk long enough for a manual count
        s = next((x for x in ctx.states.get("F_safe_attended", []) if c.end - 1.0 <= x.start <= c.end + gap), None)
        if (s is None or any(x.end >= c.start - look and x.start <= s.start + after for x in counts)
                or any(x.end >= c.start - client_s and x.start <= s.start + after for x in hands + clients)):
            continue                                   # no safe visit / counted / client operation (п. 42 exception)
        t = round(s.start + lag, 1)
        out.append(_post_cand(rule, ctx, "F_safe_attended", t, s.start, s.end))
    return out


def count_alone_open_safe(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    """X-20: the cashier takes cash out of the safe, leaves the safe open, sits at the desk and counts the cash alone.
    The same solo count after the safe was closed is labelled normal (X-21), so the safe must stand open and unattended
    during the count: every candidate lies inside an R02 situation. Excluded: the п. 42 client exception (a client
    cash flow or a handover within client_s), collection bags (X-14 opens and counts a bag alone, labelled normal), a
    cashier away from the desk or untracked, and visitors in the room for more than max_busy of the count (a short
    reflection blip is tolerated)."""
    p = rule.params
    min_s, client_s, max_busy = p.get("min_s", 2.0), p.get("client_s", 30.0), p.get("max_busy", 0.25)
    if ctx.people.cashier_id is None:
        return []
    names = p.get("attend_fact", [])
    att = [a for n in ([names] if isinstance(names, str) else names) for a in ctx.states.get(n, [])]
    flows = [x for n in ("F_cash_from_client", "F_cash_to_client", "F_handover") for x in ctx.states.get(n, [])]
    bags = ctx.states.get("F_bag_desk", [])
    out = []
    for f in p.get("count_facts", ["F_cash_hands", "F_counting"]):
        for c in ctx.states.get(f, []):
            safe = next((o for o in ctx.states.get("F_safe_open", [])
                         if o.start < c.start and o.end >= c.start + min_s), None)   # opened before the count
            ts = [t for t in ctx.people.times if c.start <= t <= c.end]
            if (c.length < min_s or safe is None or not ts
                    or any(x.end >= c.start - client_s and x.start <= c.end + client_s for x in flows)
                    or any(x.end >= c.start - client_s and x.start <= c.end for x in bags)
                    or sum(ctx.people.others_at(t) > 0 for t in ts) > max_busy * len(ts)):
                continue
            run: list[float] = []
            for t in ts + [None]:
                if (t is not None and t <= safe.end and not ctx.people.away_at(t)
                        and not any(a.start <= t <= a.end for a in att)):
                    run.append(t)                      # at the desk, safe open, nobody at the safe
                    continue
                if run and run[-1] - run[0] >= min_s:
                    out.append(_post_cand(rule, ctx, f, round(run[0], 1), run[0], c.end))
                    break
                run = []
    return out


def door_left_open(rule: Rule, ctx: RuleContext) -> list[Candidate]:
    """The door panel stays unlike the first frame for min_s (someone opened it and nobody closed it). A brief
    opening for a passage returns to the reference and does not count; unknown samples neither extend nor break a run.
    A run reaching the end of the video counts from end_min_s. Timecode: onset + lag_s (the passage itself is over)."""
    p = rule.params
    thr, min_s, end_min, lag = p.get("thr", 0.15), p.get("min_s", 8.0), p.get("end_min_s", 4.0), p.get("lag_s", 1.0)
    out, start, last = [], None, None
    for t, s in ctx.door + [(float("inf"), 0.0)]:
        if s < 0:
            continue
        if s >= thr:
            start, last = (t if start is None else start), t
            continue
        if start is not None:
            length, at_end = last - start, t == float("inf") and last >= ctx.duration - 1.0
            if length >= min_s or (at_end and length >= end_min):
                at = round(min(start + lag, last), 1)
                out.append(Candidate(rule.id, at, round(min(at + 2.0, last), 1), None, 0.8, True))
            start = None
    return out


FAMILIES: dict[str, Callable[[Rule, RuleContext], list[Candidate]]] = {
    "on_cashier_left": on_cashier_left,
    "unattended_state": unattended_state,
    "cashier_left_with_client": cashier_left_with_client,
    "on_entry": on_entry,
    "entry_during_service": entry_during_service,
    "duration": duration,
    "presence": presence,
    "quality": quality,
}
FAMILIES.update({
    "count_after_client_left": count_after_client_left,
    "handover_without_id": handover_without_id,
    "collector_quick_leave": collector_quick_leave,
    "return_after_count": return_after_count,
    "door_left_open": door_left_open,
    "state_left_unattended": state_left_unattended,
    "appeared_while_away": appeared_while_away,
    "deposit_onto_leftover": deposit_onto_leftover,
    "desk_to_safe_without_count": desk_to_safe_without_count,
    "count_alone_open_safe": count_alone_open_safe,
})


def evaluate(rules: list[Rule], ctx: RuleContext) -> list[Candidate]:
    seen: set[tuple[str, float]] = set()
    out = []
    for r in rules:
        for fam in r.families:
            for c in FAMILIES[fam](r, ctx):
                key = (c.rule_id, round(c.t, 1))
                if key not in seen:
                    seen.add(key)
                    out.append(c)
    return out
