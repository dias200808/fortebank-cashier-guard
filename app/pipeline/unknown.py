"""S8: describe event clips, then check the description against each catalog rule (describe -> match)."""
import time

from app.pipeline.frames import VideoFrames
from app.pipeline.people import PeopleTimeline
from app.pipeline.rules import Candidate, Rule
from app.pipeline.verify import clip_times
from app.pipeline.vlm import VLM

DESCRIBE = ("These are consecutive frames from a bank cashier room camera. Describe in 2-3 sentences what the people do "
            "with money, bags, documents, the safe, the door and the laptop. Only state what is visible.")
MATCH = ("Description of a bank cashier room video: \"{desc}\"\n"
         "Rule {rid}: {rtype}.\nDoes the description show a breach of this rule?")


def scan_unknown(frames: VideoFrames, people: PeopleTimeline, rules: list[Rule], vlm: VLM,
                 already: set[str], threshold: float = 0.85, max_events: int = 6,
                 deadline: float | None = None) -> list[tuple[Candidate, float]]:
    todo = [r for r in rules if r.id not in already]
    if not todo:
        return []
    best: dict[str, tuple[Candidate, float]] = {}
    for ev in people.events[:max_events]:
        if deadline is not None and time.monotonic() > deadline:
            break
        imgs = [frames.at(x) for x in clip_times(ev.t, [-2, 4], frames.duration, step=1.0, max_frames=6)]
        desc = (vlm.text(imgs, DESCRIBE, 160) or "").replace('"', "'")
        probs = vlm.yes_probs([([], MATCH.format(desc=desc, rid=r.id, rtype=r.type)) for r in todo])
        for r, p in zip(todo, probs):
            if p >= threshold and (r.id not in best or p > best[r.id][1]):
                best[r.id] = (Candidate(r.id, ev.t, ev.t, None, p, False), p)
    return list(best.values())
