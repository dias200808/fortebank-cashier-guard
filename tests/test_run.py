import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.pipeline.probes import load_facts
from app.pipeline.rules import load_rules
from app.pipeline.run import Models, analyze, build_processor, remaining_s
from app.pipeline.vlm import FakeVLM, VLMError

W, H = 96, 160


def painter(i):
    """Grey room (not dark, so the camera-covered rule stays silent). Frames 0-269 (9 s @30fps): cashier
    (red block) seated; afterwards gone. Cash (white block) on the desk from 6 s."""
    img = np.full((H, W, 3), 120, np.uint8)
    if i < 270:
        img[40:100, 30:60] = (200, 0, 0)
    if i >= 180:
        img[110:130, 10:40] = 255
    return img


class BlockDetector:
    def __call__(self, images):
        return [np.array([[30, 40, 60, 100, 0.95]], np.float32) if im[70, 45, 0] > 150 and im[70, 45, 1] < 80
                else np.zeros((0, 5), np.float32) for im in images]


def has_white(img):
    return bool((img.min(axis=2) > 200).any())


def cash_vlm():
    def answer(images, q):
        if "banknotes" in q or "money" in q.lower() or "cash" in q.lower():
            return 0.95 if images and has_white(images[0]) else 0.05   # text-only S8 match probes have no images
        return 0.05
    return FakeVLM(answer=answer, reply=lambda imgs, p: '{"bbox_2d": [0, 600, 1000, 900]}' if "desk" in p else '{"bbox_2d": null}')


SETTINGS = replace(Settings(), det_fps=5.0, time_budget_s=120.0)
RULES = load_rules(Path("app/rules.yaml"))
FACTS = load_facts(Path("app/facts.yaml"))


def test_cash_left_on_desk_is_reported(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))                 # 15 s
    verdict, violations, debug = analyze(path, Models(cash_vlm(), BlockDetector()), SETTINGS, RULES, FACTS)
    assert verdict == "violation"
    r01 = [v for v in violations if v.rule_id == "R01"]
    assert r01 and abs(r01[0].timecode_sec - 9.0) <= 1.0
    assert 1 <= len(r01[0].evidence_frames) <= 3


def test_clean_video_no_violation(tmp_video):
    path = tmp_video(150, 30, lambda i: painter(0), size=(W, H))     # cashier seated, no cash
    verdict, violations, _ = analyze(path, Models(cash_vlm(), BlockDetector()), SETTINGS, RULES, FACTS)
    assert verdict == "no_violation" and violations == []


def test_budget_skips_verification(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))
    s = replace(SETTINGS, time_budget_s=0.0)
    verdict, violations, debug = analyze(path, Models(cash_vlm(), BlockDetector()), s, RULES, FACTS)
    assert debug["verification_skipped"] is True
    assert verdict == "violation"                                    # strong probe candidates still accepted


def test_process_rejects_non_video(tmp_path):
    bad = tmp_path / "x.mov"
    bad.write_bytes(b"not a video")
    process = build_processor(SETTINGS, Models(cash_vlm(), BlockDetector()))
    with pytest.raises(Exception):
        process(bad, "job1")


def test_slow_vlm_truncates_probes_within_budget(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))
    base = cash_vlm()

    def slow(images, q):
        time.sleep(0.01)
        return base.answer(images, q)

    s = replace(SETTINGS, time_budget_s=1.0)
    t0 = time.monotonic()
    _, _, debug = analyze(path, Models(FakeVLM(answer=slow, reply=base.reply), BlockDetector()), s, RULES, FACTS)
    assert debug["probes_truncated"] is True
    assert time.monotonic() - t0 < 6.0


def test_unknown_channel_vlm_failure_keeps_cv_result(tmp_video):
    from app.pipeline.vlm import VLMError

    path = tmp_video(450, 30, painter, size=(W, H))
    base = cash_vlm()

    def down(images, p):
        raise VLMError("down")

    verdict, violations, debug = analyze(path, Models(FakeVLM(answer=base.answer, reply=down), BlockDetector()),
                                         SETTINGS, RULES, FACTS)
    assert verdict == "violation"
    assert any(v.rule_id == "R01" for v in violations)
    assert debug.get("unknown_error") == "down"


def test_remaining_s_counts_from_submission(tmp_path):
    video = tmp_path / "input.mov"
    video.write_bytes(b"x")
    now = time.time()
    (tmp_path / "submitted_at").write_text(str(now - 200.0))
    assert abs(remaining_s(video, Settings(), now=now) - 70.0) < 1e-6


def test_remaining_s_fallbacks(tmp_path):
    video = tmp_path / "input.mov"
    assert remaining_s(video, Settings()) == 270.0                      # neither file exists
    video.write_bytes(b"x")
    assert 265.0 < remaining_s(video, Settings()) <= 270.0              # video mtime


def test_zero_budget_skips_verification_and_returns(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))
    verdict, violations, debug = analyze(path, Models(cash_vlm(), BlockDetector()), SETTINGS, RULES, FACTS, budget_s=0.0)
    assert debug["verification_skipped"] is True and verdict in {"violation", "no_violation"}


def test_verification_vlm_failure_keeps_strong_cv_result(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))
    base = cash_vlm()

    def answer(images, q):
        if "consecutive video frames" in q:                             # S7 clip note
            raise VLMError("S7 down")
        return base.answer(images, q)

    verdict, violations, debug = analyze(path, Models(FakeVLM(answer=answer, reply=base.reply), BlockDetector()),
                                         SETTINGS, RULES, FACTS)
    assert debug["verify_error"] == "S7 down"
    assert any(v.rule_id == "R01" for v in violations)


def s8_vlm(match_rule="R05"):
    base = cash_vlm()

    def answer(images, q):
        if q.startswith("Description of a bank cashier room video"):
            return 0.95 if f"Rule {match_rule}:" in q else 0.05
        if "consecutive video frames" in q:
            return 0.05                                                 # S7 rejects everything
        return base.answer(images, q)

    return FakeVLM(answer=answer, reply=base.reply)


def test_s8_hit_must_pass_verification(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))
    _, violations, debug = analyze(path, Models(s8_vlm(), BlockDetector()), SETTINGS, RULES, FACTS)
    assert not any(v.rule_id == "R05" for v in violations)
    s8 = [d for d in debug["candidates_detail"] if d["source"] == "s8"]
    assert s8 and s8[0]["rule"] == "R05" and s8[0]["verified"] is False


def test_unknown_channel_can_be_switched_off(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))
    prompts = []
    base = cash_vlm()

    def reply(images, p):
        prompts.append(p)
        return base.reply(images, p)

    s = replace(SETTINGS, unknown_channel=False)
    analyze(path, Models(FakeVLM(answer=base.answer, reply=reply), BlockDetector()), s, RULES, FACTS)
    assert not any(p.startswith("These are consecutive frames") for p in prompts)


def test_candidates_detail_lists_cv_candidates(tmp_video):
    path = tmp_video(450, 30, painter, size=(W, H))
    _, _, debug = analyze(path, Models(cash_vlm(), BlockDetector()), SETTINGS, RULES, FACTS)
    r01 = [d for d in debug["candidates_detail"] if d["rule"] == "R01"]
    assert r01 and r01[0]["source"] == "cv" and r01[0]["strong"] is True
    assert set(r01[0]) == {"rule", "t", "score", "strong", "source", "verified"}


def test_coarse_first_spreads_the_dense_extras():
    """A probe stop thins every event's dense window instead of dropping the last seconds (X-4: R06/R07 at 55/68 s)."""
    from app.pipeline.run import _coarse_first
    extras = [round(e + k * 0.2, 2) for e in (5.1, 30.1, 55.1, 68.1) for k in (-2, -1, 0, 1, 2)]
    times = sorted([float(t) for t in range(0, 71)] + extras)
    order = _coarse_first(times, 1.0, 70.0)
    assert sorted(order) == times                                    # a permutation
    assert order[:36] == [float(t) for t in range(0, 71, 2)]          # 2 s grid first
    assert set(order[36:71]) == {float(t) for t in range(1, 71, 2)}   # then the rest of the 1 fps grid
    first_half = order[71:81]                                        # a stop after half of the extras
    assert all(any(abs(t - e) <= 0.45 for t in first_half) for e in (5.1, 30.1, 55.1, 68.1))
