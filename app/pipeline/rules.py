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


def load_rules(path: Path) -> list[Rule]:
    rules = []
    for d in yaml.safe_load(Path(path).read_text(encoding="utf-8")):
        fam = d.pop("family")
        d["families"] = fam if isinstance(fam, list) else [fam]
        rules.append(Rule(**d))
    return [r for r in rules if r.enabled]


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
    min_s = rule.params.get("min_s", 2.0)
    return [_post_cand(rule, ctx, f, iv.start, iv.start, iv.end)
            for f in rule.facts for iv in ctx.states.get(f, []) if iv.length >= min_s]


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
    look = rule.params.get("lookback_s", 20.0)
    ids = ctx.states.get("F_id_card", [])
    out = []
    for f in rule.facts:
        for iv in ctx.states.get(f, []):
            if ctx.people.others_at(iv.start) == 0:
                continue                               # no client at the handover -> nothing to identify
            if not any(x.end >= iv.start - look and x.start <= iv.end + 3.0 for x in ids):
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
