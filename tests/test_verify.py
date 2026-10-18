import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.geom import crop
from app.pipeline.people import PeopleTimeline
from app.pipeline.probes import Fact
from app.pipeline.rules import Candidate, Rule
from app.pipeline.scene import Zones
from app.pipeline.verify import build_items, clip_times, decide, verify_all
from app.pipeline.vlm import FakeVLM

RULE = Rule("R01", "t", ["on_cashier_left"], ["F_cash_desk"], {}, {"act": "a", "point": "p"}, "В", "{t}",
            verify=[{"q": "money A?", "window": [0, 5]}, {"q": "money B?", "window": [0, 5]},
                    {"q": "seat empty?", "window": [1, 6], "zone": "full", "expect": "yes"}])
FACTS = {"F_cash_desk": Fact("F_cash_desk", "desk", "q")}
VF = VideoFrames([i * 0.2 for i in range(100)], [np.zeros((50, 30, 3), np.uint8)] * 100, 20.0)


def test_clip_times_clamped_and_capped():
    assert clip_times(1.0, [-3, 5], 20.0, step=1.0, max_frames=8) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert len(clip_times(10.0, [-30, 30], 20.0, step=0.5, max_frames=8)) == 8


def test_decide_majority_with_expect_no():
    rule = Rule("X", "t", [], [], {}, {}, "В", "{t}", verify=[{"q": "a"}, {"q": "b"}, {"q": "c", "expect": "no"}])
    assert decide(rule, [0.9, 0.8, 0.1])[0] is True
    assert decide(rule, [0.9, 0.2, 0.9])[0] is False


def test_required_question_is_a_veto():
    rule = Rule("X", "t", [], [], {}, {}, "Н", "{t}",
                verify=[{"q": "a"}, {"q": "b"}, {"q": "safe", "required": True}, {"q": "d", "expect": "no"}])
    assert decide(rule, [0.9, 0.9, 0.9, 0.1])[0] is True
    assert decide(rule, [0.9, 0.9, 0.2, 0.1])[0] is False          # 3 of 4 agree, but the required one does not
    assert decide(rule, [0.9, 0.2, 0.9, 0.9])[0] is False          # required holds, no majority


def test_verify_all_confirms_and_rejects():
    c = Candidate("R01", 9.0, 10.0, "F_cash_desk", 0.9)
    yes = verify_all([c], {"R01": RULE}, VF, Zones(desk=(0, 0.5, 1, 1)), FACTS, FakeVLM(answer=lambda i, q: 0.9))
    no = verify_all([c], {"R01": RULE}, VF, Zones(desk=(0, 0.5, 1, 1)), FACTS, FakeVLM(answer=lambda i, q: 0.1))
    assert yes[0][1] is True and no[0][1] is False


def test_rule_without_verify_passes_through():
    r = Rule("R18", "t", ["quality"], [], {}, {}, "Н", "{t}", verify=[])
    c = Candidate("R18", 3.0, 3.0, None, 0.9, True)
    out = verify_all([c], {"R18": r}, VF, Zones(), FACTS, FakeVLM())
    assert out[0][1] is True and out[0][2] == 0.9


def test_clip_times_reaches_min_frames_on_short_window():
    ts = clip_times(10.0, [0, 4], 20.0)
    assert len(ts) == 6 and ts[0] == 10.0 and ts[-1] == 14.0


def test_mixed_batch_keeps_owner_mapping():
    r_a = Rule("A", "a", [], ["F_cash_desk"], {}, {}, "В", "{t}", verify=[{"q": "qa1"}, {"q": "qa2"}, {"q": "qa3"}])
    r_e = Rule("E", "e", [], [], {}, {}, "Н", "{t}", verify=[])
    r_b = Rule("B", "b", [], ["F_cash_desk"], {}, {}, "В", "{t}", verify=[{"q": "qb1"}, {"q": "qb2"}, {"q": "qb3"}])
    cands = [Candidate("A", 5.0, 5.0, "F_cash_desk", 0.9), Candidate("E", 6.0, 6.0, None, 0.7, True),
             Candidate("B", 7.0, 7.0, "F_cash_desk", 0.9)]
    vlm = FakeVLM(answer=lambda imgs, q: 0.9 if q.startswith("qa") else 0.1)
    out = verify_all(cands, {"A": r_a, "E": r_e, "B": r_b}, VF, Zones(desk=(0, 0.5, 1, 1)), FACTS, vlm)
    assert [(c.rule_id, ok) for c, ok, _ in out] == [("A", True), ("E", True), ("B", False)]
    assert out[1][2] == 0.7


def test_cashier_zone_clip_uses_the_cashier_box_per_frame():
    """R11's S7 clip is cropped to the cashier like its probe; frames with the cashier out of view are dropped."""
    r11 = Rule("R11", "t", ["presence"], ["F_phone"], {}, {}, "С", "{t}", verify=[{"q": "phone?", "window": [0, 4]}])
    facts = {"F_phone": Fact("F_phone", "cashier", "q")}
    times = [i * 0.2 for i in range(100)]
    boxes = [(0.0, 0.0, 0.5, 0.5) if t < 2.0 else None for t in times]
    people = PeopleTimeline(times, 1, [b is None for b in boxes], [0] * 100, [], None, boxes)
    seen = []
    vlm = FakeVLM(answer=lambda imgs, q: seen.append([im.shape[:2] for im in imgs]) or 0.9)
    out = verify_all([Candidate("R11", 0.0, 0.0, "F_phone", 0.9)], {"R11": r11}, VF, Zones(), facts, vlm, people)
    assert out[0][1] is True and seen == [[(747, 448)] * 3]        # 0.0, 0.8, 1.6 s in view; 2.4-4.0 s dropped
    verify_all([Candidate("R11", 5.0, 5.0, "F_phone", 0.9)], {"R11": r11}, VF, Zones(), facts, vlm, people)
    assert seen[-1] == [(50, 30)] * 6                               # never in view: full frames, not an empty clip


CHOICE = {"q": "What is in her hands? A: phone; B: money", "options": ["A", "B"], "expect": "A", "required": True,
          "window": [0, 4]}


def test_decide_reads_a_choice_question_as_p_of_the_expected_option():
    rule = Rule("X", "t", [], [], {}, {}, "С", "{t}", verify=[{"q": "a"}, {"q": "b"}, CHOICE])
    assert decide(rule, [0.9, 0.9, 0.8])[0] is True
    assert decide(rule, [0.9, 0.9, 0.1])[0] is False                # P(phone) 0.1: veto although both yes/no agree


def test_choice_veto_rejects_banknotes_taken_for_a_phone():
    """X-8/X-13/X-14: the per-frame phone probe and the yes/no phrasings say "phone" while the cashier counts notes;
    the clip with the option list answers B (money) and vetoes R11."""
    r11 = Rule("R11", "t", ["presence"], ["F_phone"], {}, {}, "С", "{t}",
               verify=[{"q": "phone?", "window": [0, 4]}, {"q": "smartphone?", "window": [0, 4]}, CHOICE])
    c = [Candidate("R11", 5.0, 5.0, "F_phone", 0.9)]
    facts = {"F_phone": Fact("F_phone", "full", "q")}
    asked = []

    def choice(imgs, prompt, options):
        asked.append((len(imgs), options))
        return {"A": 0.02, "B": 0.98}

    money = verify_all(c, {"R11": r11}, VF, Zones(), facts, FakeVLM(answer=lambda i, q: 0.95, choice=choice))
    phone = verify_all(c, {"R11": r11}, VF, Zones(), facts,
                       FakeVLM(answer=lambda i, q: 0.95, choice=lambda i, p, o: {"A": 0.99, "B": 0.01}))
    silent = verify_all(c, {"R11": r11}, VF, Zones(), facts, FakeVLM(answer=lambda i, q: 0.95))   # no choice answer
    assert money[0][1] is False and phone[0][1] is True and silent[0][1] is False
    assert asked == [(6, ["A", "B"])]                               # the clip, once, with the option letters


def test_scale_upscales_only_that_questions_crops():
    r = Rule("R01", "t", [], ["F_cash_desk"], {}, {}, "В", "{t}",
             verify=[{"q": "a", "window": [0, 5], "scale": 2}, {"q": "b", "window": [0, 5]}])
    (big, _), (plain, _) = build_items(Candidate("R01", 9.0, 9.0, "F_cash_desk", 0.9), r, VF, Zones(desk=(0, 0.5, 1, 1)),
                                       FACTS)
    h, w = plain[0].shape[:2]
    assert len(big) == len(plain) and big[0].shape[:2] == (2 * h, 2 * w) and plain[0].shape == crop(VF.at(9.0), (0, 0.5, 1, 1)).shape
