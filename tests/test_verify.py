import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.probes import Fact
from app.pipeline.rules import Candidate, Rule
from app.pipeline.scene import Zones
from app.pipeline.verify import clip_times, decide, verify_all
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
