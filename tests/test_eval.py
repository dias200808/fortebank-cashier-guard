from pathlib import Path

from app.schemas import Violation
from scripts.eval import candidate_recall, score
from scripts.labels import load_labels


def V(t, rid):
    return Violation(type=rid, timecode_sec=t, description="", evidence_frames=[], rule_id=rid)


def test_labels_parse_and_dedupe():
    labels = load_labels(Path("open/open_labels.xlsx"))
    assert labels["X-1"] == [22.0] and labels["X-3"] == []
    assert sorted(labels["X-4"]) == [57.0, 68.0] and "X-47" not in labels and "X-48" not in labels
    assert len(labels) == 40


def test_score_counts_recall_fp_time_and_type():
    labels = {"A": [10.0], "B": [], "C": [5.0]}
    preds = {"A": [V(10.8, "R01")], "B": [V(3.0, "R08")], "C": []}
    types = {("A", 10.0): "R01", ("C", 5.0): "R08"}
    layouts = {"A": "L1", "B": "L1", "C": "L2"}
    s = score(preds, labels, types, layouts)
    assert s["all"]["violation_videos"] == 2 and s["all"]["caught"] == 1
    assert s["all"]["clean_videos"] == 1 and s["all"]["false_alarms"] == 1
    assert s["all"]["time_within_2s"] == 1 and s["all"]["type_correct"] == 1
    assert s["L2"]["caught"] == 0


def test_type_not_credited_when_prediction_is_far():
    labels = {"Y": [10.0]}
    preds = {"Y": [V(100.0, "R01")]}
    s = score(preds, labels, {("Y", 10.0): "R01"}, {"Y": "L"})
    assert s["all"]["time_within_2s"] == 0 and s["all"]["type_correct"] == 0


def test_two_labels_on_one_video():
    labels = {"X-4": [57.0, 68.0]}
    preds = {"X-4": [V(57.5, "R06"), V(68.4, "R07")]}
    s = score(preds, labels, {("X-4", 57.0): "R06", ("X-4", 68.0): "R07"}, {"X-4": "A"})
    assert s["all"]["violation_videos"] == 1 and s["all"]["caught"] == 1
    assert s["all"]["labels"] == 2 and s["all"]["time_within_2s"] == 2 and s["all"]["type_correct"] == 2


def test_candidate_recall_counts_labels_with_a_nearby_candidate():
    details = {"A": [{"rule": "R01", "t": 10.5, "source": "cv"}, {"rule": "R08", "t": 60.0, "source": "s8"}],
               "B": [{"rule": "R05", "t": 50.0, "source": "cv"}],
               "C": []}
    labels = {"A": [10.0, 30.0], "B": [45.0], "C": [5.0], "D": [1.0], "E": []}   # D was not run
    assert candidate_recall(details, labels) == 0.25
    assert candidate_recall(details, labels, win=5.0) == 0.5
    assert candidate_recall({}, labels) is None
