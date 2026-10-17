import time

import httpx
import numpy as np

from app.pipeline import people
from app.pipeline.frames import VideoFrames
from app.pipeline.people import Track, analyze_people
from app.pipeline.roles import COLORS, MAX_MARKS, candidates, draw_marks, identify_cashier, pick_times
from app.pipeline.scene import Zones
from app.pipeline.vlm import FakeVLM, OpenAIVLM, VLMError, p_choices

T = [round(i * 0.2, 3) for i in range(100)]          # 20 s at 5 fps
H, W = 120, 64
CASHIER_BOX = (0.6, 0.3, 0.8, 0.7)                     # right of the desk
CLIENT_BOX = (0.1, 0.3, 0.3, 0.7)                      # left of the desk
DESK = (0.0, 0.4, 1.0, 0.8)


def track(tid, t0, t1, box):
    ts = [t for t in T if t0 <= t <= t1]
    return Track(tid, ts, [box] * len(ts))


def video():
    return VideoFrames(list(T), [np.full((H, W, 3), 40, np.uint8) for _ in T], 20.0)


def marked_number(img, box=CASHIER_BOX):
    """Number drawn on the box whose left edge is at box[0]: read the outline colour there."""
    h, w = img.shape[:2]
    px = tuple(int(v) for v in img[int(round((box[1] + box[3]) / 2 * (h - 1))), int(round(box[0] * (w - 1)))])
    return COLORS.index(px) + 1 if px in COLORS else 0


def test_draw_marks_numbers_left_to_right_and_maps_tracks():
    img = np.zeros((H, W, 3), np.uint8)
    out, mapping = draw_marks(img, [(7, CASHIER_BOX), (3, CLIENT_BOX)])
    assert mapping == {1: 3, 2: 7}
    assert marked_number(out, CLIENT_BOX) == 1 and marked_number(out, CASHIER_BOX) == 2
    assert img.sum() == 0 and out.shape == img.shape            # input untouched


def test_candidates_drop_short_and_tiny_tracks_and_cap():
    real = track(1, 0, 20, CASHIER_BOX)
    short = track(2, 5.0, 6.0, CLIENT_BOX)
    reflection = track(3, 0, 20, (0.0, 0.1, 0.03, 0.15))
    assert [c.id for c in candidates([real, short, reflection])] == [1]
    many = [track(i, 0, 20, CASHIER_BOX) for i in range(12)]
    assert len(candidates(many)) == MAX_MARKS


def test_pick_times_prefers_multi_person_and_adds_single_at_desk():
    cashier, client = track(1, 0, 20, CASHIER_BOX), track(2, 3.0, 9.0, CLIENT_BOX)
    ts = pick_times(video(), [cashier, client], DESK)
    assert 1 <= len(ts) <= 3
    assert any(3.0 <= t <= 9.0 for t in ts)                    # both people
    assert any(not 3.0 <= t <= 9.0 for t in ts)                # cashier alone at the desk


def scenario():
    """Client sits at the desk the whole clip; the cashier arrives at 2 s and leaves at 12 s."""
    return [track(1, 2.0, 12.0, CASHIER_BOX), track(2, 0, 20, CLIENT_BOX)]


def test_identify_cashier_picks_marked_track_from_digit_probs():
    def choice(imgs, prompt, options):
        n = str(marked_number(imgs[0]))
        return {o: (0.9 if o == n else 0.1 / (len(options) - 1)) for o in options}

    vlm = FakeVLM(choice=choice)
    tid, conf, dbg = identify_cashier(video(), scenario(), Zones(desk=DESK), vlm)
    assert tid == 1 and conf >= 0.6 and dbg["method"] == "choice_probs" and 1 <= vlm.calls <= 3
    assert all(set(f["marks"].values()) <= {1, 2} for f in dbg["frames"])


def test_identify_cashier_costs_at_most_three_requests():
    crowd = [track(i, 0, 20, (0.1 * i, 0.3, 0.1 * i + 0.08, 0.7)) for i in range(9)]
    vlm = FakeVLM(choice=lambda imgs, p, opts: {o: float(o == "4") for o in opts})
    tid, _, dbg = identify_cashier(video(), crowd, Zones(desk=DESK), vlm)
    assert vlm.calls <= 3 and tid == 3 and len(dbg["frames"]) <= 3        # number 4 = 4th box from the left


def test_identify_cashier_without_choice_probs_parses_text_reply():
    class TextOnly:                                                        # duck-typed VLM: text() only
        calls = 0

        def text(self, imgs, prompt, max_tokens=256):
            TextOnly.calls += 1
            return f"{marked_number(imgs[0])}."

    tid, conf, dbg = identify_cashier(video(), scenario(), Zones(desk=DESK), TextOnly())
    assert tid == 1 and conf == 1.0 and dbg["method"] == "text" and TextOnly.calls <= 3


def test_identify_cashier_non_numeric_answers_abstain_without_extra_requests():
    vlm = FakeVLM(answer=lambda i, q: 0.99)                               # no choice callable: choice_probs -> None
    tid, conf, dbg = identify_cashier(video(), scenario(), Zones(desk=DESK), vlm)
    assert tid is None and conf == 0.0 and dbg["reason"] == "no numeric answer" and vlm.calls == 0
    assert all(f["abstain"] for f in dbg["frames"])


def test_identify_cashier_low_confidence_gives_none():
    uniform = FakeVLM(choice=lambda imgs, p, opts: {o: 1 / len(opts) for o in opts})
    tid, conf, dbg = identify_cashier(video(), scenario(), Zones(desk=DESK), uniform)
    assert tid is None and conf < 0.6 and dbg["reason"] == "low confidence"
    none = FakeVLM(choice=lambda imgs, p, opts: {o: float(o == "0") for o in opts})
    assert identify_cashier(video(), scenario(), Zones(desk=DESK), none)[0] is None


def test_identify_cashier_survives_vlm_errors():
    def boom(*a):
        raise VLMError("down")

    for vlm in (FakeVLM(choice=boom), FakeVLM(answer=boom), object()):
        tid, conf, dbg = identify_cashier(video(), scenario(), Zones(desk=DESK), vlm)
        assert tid is None and conf == 0.0 and "reason" in dbg
    assert identify_cashier(video(), [], Zones(), FakeVLM())[0] is None
    assert identify_cashier(video(), scenario(), None, None)[0] is None


def test_identify_cashier_hung_vlm_is_abandoned_after_budget():
    def slow(imgs, p, opts):
        time.sleep(2.0)
        return {o: float(o == "1") for o in opts}

    t0 = time.monotonic()
    tid, conf, dbg = identify_cashier(video(), scenario(), Zones(desk=DESK), FakeVLM(choice=slow), budget_s=0.2)
    assert time.monotonic() - t0 < 1.5
    assert tid is None and dbg["reason"] == "vlm failed" and "TimeoutError" in dbg["errors"][0]


def test_p_choices_merges_spaced_digit_tokens_and_needs_mass():
    top = [{"token": "2", "logprob": np.log(0.6)}, {"token": " 2", "logprob": np.log(0.1)},
           {"token": "1", "logprob": np.log(0.2)}, {"token": "The", "logprob": np.log(0.1)}]
    p = p_choices(top, ["0", "1", "2"])
    assert abs(p["2"] - 0.7 / 0.9) < 1e-6 and abs(p["1"] - 0.2 / 0.9) < 1e-6 and p["0"] == 0.0
    assert p_choices([{"token": "The", "logprob": np.log(0.8)}, {"token": "1", "logprob": np.log(0.2)}],
                     ["0", "1"], min_mass=0.5) is None
    assert p_choices([{"token": "Yes", "logprob": 0.0}], ["0", "1"]) is None


def _vlm(body):
    return OpenAIVLM("http://x/v1", "m", client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body))))


def test_openai_choice_probs_logprobs_text_and_none():
    img = np.zeros((20, 10, 3), np.uint8)
    top = [{"token": "1", "logprob": np.log(0.8)}, {"token": "0", "logprob": np.log(0.2)}]
    with_lp = {"choices": [{"message": {"content": "1"}, "logprobs": {"content": [{"token": "1", "top_logprobs": top}]}}]}
    assert abs(_vlm(with_lp).choice_probs([img], "q", ["0", "1", "2"])["1"] - 0.8) < 1e-6
    no_lp = {"choices": [{"message": {"content": " 2."}, "logprobs": None}]}
    assert _vlm(no_lp).choice_probs([img], "q", ["0", "1", "2"]) == {"0": 0.0, "1": 0.0, "2": 1.0}
    words = {"choices": [{"message": {"content": "The"}}]}
    assert _vlm(words).choice_probs([img], "q", ["0", "1", "2"]) is None


def test_cashier_hint_overrides_desk_time_heuristic():
    tracks = scenario()
    plain = analyze_people(tracks, T, DESK, None, 20.0, 0.55)
    assert plain.cashier_id == 2 and not [e for e in plain.events if e.kind == "cashier_left"]
    hinted = analyze_people(tracks, T, DESK, None, 20.0, 0.55, cashier_hint=1)
    assert hinted.cashier_id == 1
    kinds = {e.kind: e.t for e in hinted.events}
    assert abs(kinds["cashier_left"] - 12.2) <= 0.4 and kinds["cashier_entered"] == 2.0
    assert hinted.others_at(15.0) == 1 and not [e for e in hinted.events if e.track_id == 2]
    assert analyze_people(tracks, T, DESK, None, 20.0, 0.55, cashier_hint=99).cashier_id == 2   # unknown id


def walker(h, shift):
    """Cashier of box height h (far camera: small, near camera: large) shifts its centre by `shift` x h at 8 s
    and is gone at 12 s."""
    w = h * 0.5
    seat = (0.5 - w / 2, 0.5 - h / 2, 0.5 + w / 2, 0.5 + h / 2)
    d = shift * h
    moved = (seat[0], seat[1] + d, seat[2], seat[3] + d)
    ts = [t for t in T if t <= 12.0]
    return Track(1, ts, [seat if t < 8.0 else moved for t in ts])


def left_at(tr):
    tl = analyze_people([tr], T, None, None, 20.0, 1.0)
    return [round(e.t, 1) for e in tl.events if e.kind == "cashier_left"]


def test_away_threshold_scales_with_cashier_box():
    for h in (0.1, 0.6):                                       # far and near camera
        assert left_at(walker(h, 2.0 * people.AWAY_K)) == [8.0]      # real move: away from 8 s
        assert left_at(walker(h, 0.5 * people.AWAY_K)) == [12.4]     # fidgeting in the chair: leaves when the track ends


def test_duplicate_box_inside_longer_track_is_not_marked():
    cashier = track(1, 0, 20, CASHIER_BOX)
    dup = track(2, 8.0, 12.0, (0.62, 0.35, 0.78, 0.6))        # second detection on the same person
    client = track(3, 0, 20, CLIENT_BOX)
    from app.pipeline.roles import visible
    assert [i for i, _ in visible([dup, cashier, client], 10.0)] == [1, 3]
    seen = []

    def choice(imgs, prompt, options):
        seen.append(len(options) - 1)
        return {o: (0.9 if o == str(marked_number(imgs[0])) else 0.05) for o in options}

    tid, _, dbg = identify_cashier(video(), [cashier, dup, client], Zones(desk=DESK), FakeVLM(choice=choice))
    assert tid == 1 and all(2 not in f["marks"].values() for f in dbg["frames"]) and max(seen) == 2
