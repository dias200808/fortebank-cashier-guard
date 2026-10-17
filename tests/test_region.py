import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.probes import Fact
from app.pipeline.report import build_violations, region_for
from app.pipeline.rules import Candidate, Rule
from app.pipeline.scene import DEFAULT_DESK, Zones
from app.schemas import JobResult, Violation

VF = VideoFrames([i * 0.2 for i in range(100)], [np.full((60, 40, 3), 100, np.uint8)] * 100, 20.0)
FACTS = {f.id: f for f in (Fact("F_phone", "full", "q"), Fact("F_cash_desk", "desk", "q"),
                           Fact("F_safe_open", "safe", "q"), Fact("F_door_open", "door", "q"))}
ZONES = Zones(desk=(0.1, 0.5, 0.9, 0.8), safe=(0.61234, 0.2, 0.95, 0.55), door=None)
NORM = {"act": "Правила № 231", "point": "п. 20"}


def rule(rid: str, facts: list[str]) -> Rule:
    return Rule(rid, "Тип", ["duration"], facts, {}, NORM, "В", "В {t} с (кадр {te} с).")


def test_region_from_primary_fact_zone():
    assert region_for(rule("R02", ["F_safe_open"]), FACTS, ZONES) == [0.612, 0.2, 0.95, 0.55]


def test_region_skips_full_frame_facts_to_first_named_zone():
    assert region_for(rule("R20", ["F_phone", "F_cash_desk"]), FACTS, ZONES) == [0.1, 0.5, 0.9, 0.8]


def test_region_none_for_full_frame_only_and_people_only_rules():
    assert region_for(rule("R11", ["F_phone"]), FACTS, ZONES) is None
    assert region_for(rule("R05", []), FACTS, ZONES) is None


def test_region_none_when_zone_missing_or_default_desk():
    assert region_for(rule("R08", ["F_door_open"]), FACTS, ZONES) is None
    assert region_for(rule("R06", ["F_cash_desk"]), FACTS, Zones(desk=DEFAULT_DESK)) is None
    assert region_for(rule("R02", ["F_safe_open"]), FACTS, None) is None
    assert region_for(rule("R02", ["F_unknown"]), FACTS, ZONES) is None


def test_region_for_fact_less_rule_from_family_zone():
    door = Zones(desk=DEFAULT_DESK, door=(0.02, 0.05, 0.4, 0.5))
    assert region_for(Rule("R08", "Т", ["door_left_open"], [], {}, NORM, "В", "{t}"), FACTS, door) == [0.02, 0.05, 0.4, 0.5]
    assert region_for(Rule("R05", "Т", ["cashier_left_with_client"], [], {}, NORM, "В", "{t}"), FACTS, ZONES) == [0.1, 0.5, 0.9, 0.8]
    assert region_for(Rule("R05", "Т", ["cashier_left_with_client"], [], {}, NORM, "В", "{t}"), FACTS, door) is None
    assert region_for(Rule("R08", "Т", ["door_left_open"], [], {}, NORM, "В", "{t}"), FACTS, ZONES) is None


def test_build_violations_sets_region():
    rules = {"R02": rule("R02", ["F_safe_open"]), "R11": rule("R11", ["F_phone"])}
    acc = [(Candidate("R02", 4.0, 5.0, "F_safe_open", 0.9), 0.9), (Candidate("R11", 12.0, 12.0, "F_phone", 0.9), 0.8)]
    by_rule = {v.rule_id: v for v in build_violations(acc, rules, VF, zones=ZONES, facts_by_id=FACTS)}
    assert by_rule["R02"].region == [0.612, 0.2, 0.95, 0.55]
    assert by_rule["R11"].region is None


def test_build_violations_without_zones_keeps_old_behaviour():
    v = build_violations([(Candidate("R02", 4.0, 5.0, "F_safe_open", 0.9), 0.9)], {"R02": rule("R02", ["F_safe_open"])}, VF)
    assert v[0].region is None


def test_schema_accepts_violation_without_region():
    v = Violation.model_validate({"type": "t", "timecode_sec": 1.0, "description": "d", "evidence_frames": []})
    assert v.region is None
    assert "region" not in Violation.model_json_schema()["required"]
    d = JobResult(job_id="a", status="done", verdict="violation", violations=[v]).to_api()
    assert "region" not in d["violations"][0]


def test_api_output_carries_region_when_set():
    v = Violation(type="t", timecode_sec=1.0, description="d", evidence_frames=[], region=[0.1, 0.2, 0.3, 0.4])
    d = JobResult(job_id="a", status="done", verdict="violation", violations=[v]).to_api()
    assert d["violations"][0]["region"] == [0.1, 0.2, 0.3, 0.4]


def test_demo_violations_carry_valid_regions(tmp_video):
    from app.demo import demo_process
    r = demo_process(tmp_video(30, 10, lambda i: np.full((96, 160, 3), i * 5, np.uint8)), "j", delay_s=0)
    for v in r.violations:
        x1, y1, x2, y2 = v.region
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
