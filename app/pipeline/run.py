"""Orchestrator: S0..S9 with a time budget."""
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.jobs import ProcessFn
from app.pipeline.change import change_times, dense_times, door_open_series, zone_change
from app.pipeline.frames import VideoFrames, load_frames
from app.pipeline.objects import ObjectEvent, ObjectTimeline, OwlDetector, detect_timeline, key_times, object_events
from app.pipeline.people import PeopleTimeline, Track, TorchvisionPersonDetector, analyze_people, track_people
from app.pipeline.probes import Fact, load_facts, probe_times, run_probes
from app.pipeline.report import build_violations
from app.pipeline.roles import identify_cashier
from app.pipeline.rules import Candidate, Rule, RuleContext, evaluate, load_rules
from app.pipeline.scene import Zones, locate_zones
from app.pipeline.timeline import intervals
from app.pipeline.verify import verify_all
from app.pipeline.vlm import VLM, FakeVLM, OpenAIVLM, VLMError
from app.schemas import JobResult, Violation

APP_DIR = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)
PROBE_CHUNK, PROBE_BUDGET_FRAC = 8, 0.6
VERIFY_CHUNK = 8                          # S7 candidates per VLM batch; the budget is checked between chunks
JOB_MARGIN_S = 20.0                       # job limit minus this = analysis budget (S9 + result write + slack)
SCENE_THR, HAND_THR = 0.03, 0.05          # probe 2026-10-17; recalibrate in Task 16
OBJ_START_FRAC, OBJ_STOP_FRAC = 0.15, 0.25  # S2b starts only before 15 % of the budget and stops at 25 %
OBJ_BATCH = 4                             # 4 x 960^2 (~14k ViT tokens) already fills the 3090; 8 only doubles the peak


@dataclass
class Models:
    vlm: VLM
    detector: object
    objects: object | None = None             # OwlDetector when OBJ_DETECTOR=owlv2


def _build_objects(settings: Settings):
    """OWLv2 is optional: a missing package or missing weights log the error and the service runs without it."""
    if settings.obj_detector.lower() != "owlv2":
        return None
    try:
        return OwlDetector(settings.obj_model, settings.device, settings.obj_threshold)
    except Exception:
        log.exception("object detector %s disabled", settings.obj_model)
        return None


def build_models(settings: Settings) -> Models:
    vlm = FakeVLM() if settings.vlm_backend == "fake" else OpenAIVLM(settings.vlm_url, settings.vlm_model, settings.vlm_workers)
    return Models(vlm, TorchvisionPersonDetector(settings.detector, settings.device), _build_objects(settings))


def _coarse_first(times: list[float], fps: float, duration: float) -> list[float]:
    """Order probe times so an early stop still leaves a usable coarse timeline: 2 s grid, rest of base grid, dense extras."""
    grid = {round(i / fps, 2) for i in range(int(duration * fps) + 1)}
    base = [t for t in times if t in grid]
    coarse = base[::2]
    keep = set(coarse)
    return coarse + [t for t in base if t not in keep] + [t for t in times if t not in grid]


def remaining_s(video_path: Path, settings: Settings, now: float | None = None) -> float:
    """Seconds left of settings.job_limit_s, counted from the job's submission (queue wait included)."""
    now = time.time() if now is None else now
    try:
        submitted = float((video_path.parent / "submitted_at").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        try:
            submitted = video_path.stat().st_mtime
        except OSError:
            return settings.job_limit_s
    return settings.job_limit_s - (now - submitted)


Checked = list[tuple[Candidate, bool, float]]


def _verify_chunked(cands: list[Candidate], rules_by_id: dict[str, Rule], frames: VideoFrames, zones: Zones,
                    facts_by_id: dict[str, Fact], vlm: VLM, t0: float, budget: float, debug: dict) -> Checked:
    """S7 in chunks: over budget or VLM failure -> the rest fall back to (c, c.strong, c.score)."""
    out: Checked = []
    for i in range(0, len(cands), VERIFY_CHUNK):
        if time.monotonic() - t0 > budget:
            debug["verification_truncated"] = True
            return out + [(c, c.strong, c.score) for c in cands[i:]]
        try:
            out += verify_all(cands[i:i + VERIFY_CHUNK], rules_by_id, frames, zones, facts_by_id, vlm)
        except VLMError as e:
            log.warning("verification stopped: %s", e)
            debug["verify_error"] = str(e)
            return out + [(c, c.strong, c.score) for c in cands[i:]]
    return out


def _objects(frames: VideoFrames, models: Models, settings: Settings, zones: Zones, people: PeopleTimeline,
             tracks: list[Track], extra: list[float], t0: float, budget: float,
             debug: dict) -> tuple[ObjectTimeline | None, list[ObjectEvent]]:
    """S2b: object boxes on key frames -> timeline + events (extra probe times). Never fails the analysis."""
    if models.objects is None:
        return None, []
    if time.monotonic() - t0 > OBJ_START_FRAC * budget:
        debug["objects_skipped"] = True
        return None, []
    t1 = time.monotonic()
    try:
        times = key_times(frames, settings.obj_fps, extra)
        tl, debug["objects_truncated"] = detect_timeline(frames, models.objects, times, OBJ_BATCH,
                                                         deadline=t0 + OBJ_STOP_FRAC * budget)
        events = object_events(tl, zones.desk, zones.safe, people, tracks)
    except Exception as e:
        log.exception("object stage skipped")
        debug["objects_error"] = repr(e)[:200]
        return None, []
    debug.update(obj_frames=len(tl.times), obj_events=[e.__dict__ for e in events],
                 t_objects=round(time.monotonic() - t1, 1))
    return tl, events


def _detail(c: Candidate, ok: bool, score: float, source: str) -> dict:
    return {"rule": c.rule_id, "t": c.t, "score": round(score, 3), "strong": c.strong, "source": source, "verified": ok}


def analyze(path: Path, models: Models, settings: Settings, rules: list[Rule], facts: list[Fact],
            budget_s: float | None = None) -> tuple[str, list[Violation], dict]:
    t0 = time.monotonic()
    b = settings.time_budget_s if budget_s is None else min(settings.time_budget_s, budget_s)
    debug: dict = {"verification_skipped": False, "probes_truncated": False, "budget_s": round(b, 1)}
    frames = load_frames(path, settings.det_fps)
    if not frames.images:
        raise ValueError("no video frames could be decoded")
    h, w = frames.images[0].shape[:2]
    zones = locate_zones(frames.at(min(1.0, frames.duration / 2)), models.vlm, settings.vlm_coord_mode)
    tracks = track_people(frames, models.detector, settings.det_fps, batch=settings.det_batch)
    # Qwen picks the cashier among numbered person boxes (angle-independent); None keeps the desk-time heuristic
    cashier_hint, _, debug["roles"] = identify_cashier(frames, tracks, zones, models.vlm)
    people = analyze_people(tracks, frames.times, zones.desk, zones.safe, frames.duration, w / h,
                            cashier_hint=cashier_hint)

    changes = [change_times(zone_change(frames, box, tracks, mask_people=True), SCENE_THR)
               for box in (zones.desk, zones.safe, zones.door)]
    objects, obj_events = _objects(frames, models, settings, zones, people, tracks,
                                   [e.t for e in people.events] + [t for ch in changes for t in ch], t0, b, debug)
    extra = [e.t for e in people.events] + dense_times([e.t for e in obj_events])
    for ch in changes:
        extra += dense_times(ch)
    extra += dense_times(change_times(zone_change(frames, zones.desk, tracks, mask_people=False), HAND_THR))
    times = _coarse_first(probe_times(frames.duration, settings.probe_fps, extra, settings.max_probe_frames),
                          settings.probe_fps, frames.duration)
    series: dict[str, list] = {f.id: [] for f in facts}
    probed = 0
    for i in range(0, len(times), PROBE_CHUNK):
        if i and time.monotonic() - t0 > PROBE_BUDGET_FRAC * b:
            debug["probes_truncated"] = True
            break
        chunk = times[i:i + PROBE_CHUNK]
        for fid, pts in run_probes(frames, zones, models.vlm, facts, chunk).items():
            series[fid] += pts
        probed += len(chunk)
    series = {fid: sorted(pts) for fid, pts in series.items()}
    states = {fid: intervals(s) for fid, s in series.items()}
    debug.update(zones=zones.__dict__, events=[e.__dict__ for e in people.events], probe_frames=probed,
                 t_probes=round(time.monotonic() - t0, 1))

    quality = list(zip(frames.times, frames.brightness, frames.sharpness))
    cands = evaluate(rules, RuleContext(people, states, series, quality, frames.duration, settings.probe_fps,
                                         door_open_series(frames, zones.door, tracks)))
    rules_by_id = {r.id: r for r in rules}
    facts_by_id = {f.id: f for f in facts}
    if time.monotonic() - t0 > b:
        debug["verification_skipped"] = True
        checked = [(c, c.strong, c.score) for c in cands]
    else:
        checked = _verify_chunked(cands, rules_by_id, frames, zones, facts_by_id, models.vlm, t0, b, debug)
    accepted = [(c, score) for c, ok, score in checked if ok or c.strong]
    details = [_detail(c, ok, score, "cv") for c, ok, score in checked]
    if settings.unknown_channel and time.monotonic() - t0 < b * 0.7:
        from app.pipeline.unknown import scan_unknown
        proposed = {c.rule_id for c in cands}
        hits: list[tuple[Candidate, float]] = []
        try:
            hits = scan_unknown(frames, people, rules, models.vlm, proposed | {c.rule_id for c, _ in accepted},
                                deadline=t0 + 0.9 * b)
        except VLMError as e:
            log.warning("unknown-type channel skipped: %s", e)
            debug["unknown_error"] = str(e)
        s8 = _verify_chunked([c for c, _ in hits], rules_by_id, frames, zones, facts_by_id, models.vlm, t0, b, debug)
        accepted += [(c, score) for c, ok, score in s8 if ok]          # S8 hits are never strong: S7 must confirm
        details += [_detail(c, ok, score, "s8") for c, ok, score in s8]
    violations = build_violations(accepted, rules_by_id, frames, zones=zones, facts_by_id=facts_by_id, objects=objects)
    debug.update(candidates=len(cands), accepted=len(accepted), candidates_detail=details,
                 t_total=round(time.monotonic() - t0, 1))
    return ("violation" if violations else "no_violation"), violations, debug


def build_processor(settings: Settings, models: Models | None = None) -> ProcessFn:
    models = models or build_models(settings)
    rules = load_rules(APP_DIR / "rules.yaml")
    facts = load_facts(APP_DIR / "facts.yaml")

    def process(path: Path, job_id: str) -> JobResult:
        budget = max(0.0, remaining_s(path, settings) - JOB_MARGIN_S)
        verdict, violations, debug = analyze(path, models, settings, rules, facts, budget_s=budget)
        log.info("job %s: %s", job_id, {k: debug.get(k) for k in (
            "budget_s", "t_objects", "t_probes", "t_total", "probe_frames", "probes_truncated", "verification_skipped",
            "verification_truncated", "verify_error", "unknown_error", "objects_error", "candidates", "accepted")})
        return JobResult(job_id=job_id, status="done", verdict=verdict, violations=violations)

    return process
