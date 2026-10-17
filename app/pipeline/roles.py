"""Cashier role (between S2 tracking and analyze_people): which tracked person is the cashier, asked of the VLM
with set-of-mark boxes.

The desk-time heuristic in people.py assumes the cashier is the person who stays longest near the desk box; on an
unseen camera angle (or when a client sits at the desk longer than the cashier) that breaks. Here the VLM sees key
frames with every candidate drawn as a numbered box and names the employee, so no room layout is assumed.
"""
import logging
import statistics as st
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait

import cv2
import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import Box, center, contains, expand
from app.pipeline.people import MIN_VISIT_S, Track
from app.pipeline.scene import Zones

log = logging.getLogger(__name__)

MAX_MARKS = 9        # answers stay single digits (one token each)
MAX_FRAMES = 3
CAND_REL_H = 0.3     # candidates: median box height >= this x the tallest long track (drops glass reflections)
MIN_CONF = 0.6       # mean P(chosen) over the key frames it is marked in
MIN_GAP_S = 1.0      # key frames at least this far apart
DUP_OVERLAP = 0.7    # a box this much inside a longer track's box is a duplicate detection of that person
VIEW_PX = 896        # to_data_url downsizes to this max side; marks are sized for it
BUDGET_S = 15.0      # wall-clock cap: a hung VLM (30 s timeout x 3 attempts per call) must not stall the analysis
COLORS = [(255, 230, 0), (0, 230, 255), (255, 60, 220), (90, 255, 90), (255, 150, 30),
          (255, 255, 255), (255, 90, 90), (150, 170, 255), (200, 255, 0)]
_CUES = ("The teller uses the computer: the laptop screen faces the teller, and the safe and the banknote counting "
         "machine are on the teller's side. Clients sit or stand across the desk from the teller and see the back of "
         "the laptop screen. The teller may be seen from behind. ")
# prompt picked on 72 live key frames (scratchpad prompt_eval): 0.96 top-1 vs 0.73 for "behind the desk" wording,
# which Qwen reads from the camera's side and so picks the client when the camera looks over the cashier's back
PROMPT = ("This is a CCTV frame of a bank teller's cash office. Each person is marked with a numbered box. One of them "
          "may be the bank teller (cashier) who works here; others are clients or visitors. " + _CUES +
          "Which number is the teller? Answer with the number only, or 0 if no marked person is the teller.")


def candidates(tracks: list[Track]) -> list[Track]:
    """Real people only: visible >= MIN_VISIT_S and not much smaller than the tallest such track; at most MAX_MARKS."""
    long = [tr for tr in tracks if tr.end - tr.start >= MIN_VISIT_S]
    if not long:
        return []
    tallest = max(tr.height() for tr in long)
    big = [tr for tr in long if tr.height() >= CAND_REL_H * tallest]
    return sorted(big, key=lambda tr: (-len(tr.times), tr.id))[:MAX_MARKS]


def _overlap(a: Box, b: Box) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 1.0


def visible(cands: list[Track], t: float) -> list[tuple[int, Box]]:
    """Candidate boxes at t, longest track first; a box mostly inside a longer track's box is a duplicate detection
    of the same person (ByteTrack spawns these when someone sits down or bends) and is not marked."""
    out: list[tuple[int, Box]] = []
    for c in sorted(cands, key=lambda c: (-len(c.times), c.id)):
        b = c.box_at(t)
        if b is not None and all(_overlap(b, ob) < DUP_OVERLAP for _, ob in out):
            out.append((c.id, b))
    return out


def pick_times(frames: VideoFrames, cands: list[Track], desk: Box | None, n: int = MAX_FRAMES) -> list[float]:
    """Key frames: the one with most candidates at once, one with a single person at the desk, then frames that show
    candidates not covered yet (or a second multi-person view)."""
    edge = min(1.0, frames.duration * 0.1)
    ts = [t for t in frames.times if edge <= t <= frames.duration - edge] or list(frames.times)
    vis = {t: frozenset(i for i, _ in visible(cands, t)) for t in ts}
    near = expand(desk, 1.4) if desk else None
    by_id = {c.id: c for c in cands}
    sharp = dict(zip(frames.times, frames.sharpness)) if frames.sharpness else {}
    chosen: list[float] = []
    covered: set[int] = set()

    def gap(t: float) -> float:
        return min((abs(t - c) for c in chosen), default=1e9)

    def at_desk(t: float) -> int:
        if near is None:
            return len(vis[t])
        return sum(contains(near, *center(by_id[i].box_at(t))) for i in vis[t])

    def add(pool: list[float]) -> None:
        pool = [t for t in pool if vis[t] and gap(t) >= MIN_GAP_S]
        if pool and len(chosen) < n:
            t = max(pool, key=lambda t: (len(vis[t] - covered), len(vis[t]), min(gap(t), 5.0), sharp.get(t, 0.0)))
            chosen.append(t)
            covered.update(vis[t])

    multi = [t for t in ts if len(vis[t]) >= 2]
    add(multi)
    add([t for t in ts if at_desk(t) == 1] or [t for t in ts if len(vis[t]) == 1])
    while len(chosen) < n and {c.id for c in cands} - covered:
        before = len(chosen)
        add([t for t in ts if vis[t] - covered])
        if len(chosen) == before:
            break
    add(multi)
    add(ts)
    return sorted(chosen)


def draw_marks(image: np.ndarray, boxes: list[tuple[int, Box]]) -> tuple[np.ndarray, dict[int, int]]:
    """Draw numbered boxes (dark halo + bright outline + filled number tag), numbered left to right.
    Returns the annotated copy and {number: track_id}."""
    img = np.ascontiguousarray(image.copy())
    h, w = img.shape[:2]
    s = max(h, w) / VIEW_PX
    thick = max(2, round(4 * s))
    font, scale, tt = cv2.FONT_HERSHEY_SIMPLEX, 1.3 * s, max(2, round(3 * s))
    pad = max(3, round(6 * s))
    mapping: dict[int, int] = {}
    px = []
    for n, (tid, b) in enumerate(sorted(boxes, key=lambda kv: (center(kv[1])[0], kv[0])), 1):
        mapping[n] = tid
        px.append((n, int(round(b[0] * (w - 1))), int(round(b[1] * (h - 1))),
                   int(round(b[2] * (w - 1))), int(round(b[3] * (h - 1)))))
    for n, x1, y1, x2, y2 in px:                     # all outlines first, so no outline crosses a number tag
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thick * 2 + 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[(n - 1) % len(COLORS)], thick)
    for n, x1, y1, _, _ in px:
        label = str(n)
        (tw, th), _ = cv2.getTextSize(label, font, scale, tt)
        bw, bh = tw + 2 * pad, th + 2 * pad
        lx = min(max(0, x1), max(0, w - bw))
        ly = y1 - bh if y1 - bh >= 0 else min(y1, max(0, h - bh))
        cv2.rectangle(img, (lx, ly), (lx + bw, ly + bh), COLORS[(n - 1) % len(COLORS)], -1)
        cv2.rectangle(img, (lx, ly), (lx + bw, ly + bh), (0, 0, 0), max(1, thick // 2))
        cv2.putText(img, label, (lx + pad, ly + pad + th), font, scale, (0, 0, 0), tt, cv2.LINE_AA)
    return img, mapping


def _ask(vlm, img: np.ndarray, mapping: dict[int, int]) -> tuple[dict[int, float], float, str] | None:
    """P(track is the cashier) for each marked track and P(none) in one frame, from exactly one VLM request;
    None when the answer is not one of the numbers (the frame abstains)."""
    options = [str(i) for i in range(len(mapping) + 1)]
    fn = getattr(vlm, "choice_probs", None)
    if fn is not None:
        probs, method = fn([img], PROMPT, options), "choice_probs"
    else:                                        # VLM without choice_probs: parse the reply, still one request
        reply = (vlm.text([img], PROMPT, 2) or "").strip().strip(".")
        probs, method = ({o: float(o == reply) for o in options} if reply in options else None), "text"
    if not probs:
        return None
    return {mapping[int(o)]: p for o, p in probs.items() if o != "0"}, probs.get("0", 0.0), method


def identify_cashier(frames: VideoFrames, tracks: list[Track], zones: Zones | None, vlm,
                     budget_s: float = BUDGET_S) -> tuple[int | None, float, dict]:
    """(cashier track id or None, confidence, debug). At most MAX_FRAMES VLM requests, run in parallel and abandoned
    after budget_s. Never raises: VLM or other errors give None (the caller keeps the desk-time heuristic)."""
    debug: dict = {"candidates": [], "frames": [], "method": None, "chosen": None}
    try:
        cands = candidates(tracks)
        debug["candidates"] = [c.id for c in cands]
        if not cands or not frames.images:
            debug["reason"] = "no candidates"
            return None, 0.0, debug
        jobs = []
        for t in pick_times(frames, cands, zones.desk if zones else None):
            img, mapping = draw_marks(frames.at(t), visible(cands, t))
            jobs.append((t, img, mapping))
        if not jobs:
            debug["reason"] = "no key frames"
            return None, 0.0, debug

        def ask(job):
            try:
                return _ask(vlm, job[1], job[2])
            except Exception as e:  # noqa: BLE001 - one bad frame must not sink the others
                return e

        ex = ThreadPoolExecutor(len(jobs))
        try:
            futs = [ex.submit(ask, job) for job in jobs]
            done, _ = wait(futs, timeout=budget_s)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)   # a hung request finishes in the background
        answers = [f.result() if f in done else TimeoutError(f"no answer within {budget_s:g} s") for f in futs]
        marked: dict[int, list[float]] = {}
        votes: Counter = Counter()
        errors = []
        for (t, _, mapping), ans in zip(jobs, answers):
            if isinstance(ans, Exception):
                errors.append(f"{type(ans).__name__}: {ans}")
                continue
            if ans is None:
                debug["frames"].append({"t": t, "marks": mapping, "abstain": True})
                continue
            dist, none, method = ans
            debug["method"] = method
            for tid in mapping.values():
                marked.setdefault(tid, []).append(dist.get(tid, 0.0))
            best = max(dist, key=dist.get) if dist else None
            if best is not None and dist[best] > none:
                votes[best] += 1
            debug["frames"].append({"t": t, "marks": mapping, "p": {k: round(v, 3) for k, v in dist.items()},
                                    "none": round(none, 3)})
        if errors:
            debug["errors"] = errors[:3]
        if not marked:
            debug["reason"] = "vlm failed" if errors else "no numeric answer"
            return None, 0.0, debug
        win = max(marked, key=lambda tid: (votes[tid], sum(marked[tid]), -tid))
        conf = round(st.mean(marked[win]), 3)
        debug.update(best=win, conf=conf, votes=votes[win], marked=len(marked[win]))
        if conf >= MIN_CONF and votes[win] * 2 >= len(marked[win]):
            debug["chosen"] = win
            return win, conf, debug
        debug["reason"] = "low confidence"
        return None, conf, debug
    except Exception as e:  # noqa: BLE001 - the cashier hint is optional; the heuristic still runs
        log.warning("identify_cashier failed: %s", e)
        debug["reason"] = f"error: {type(e).__name__}: {e}"
        return None, 0.0, debug
