from app.schemas import EvidenceFrame, JobResult, RuleRef, Violation


def test_done_result_matches_tz_schema():
    r = JobResult(job_id="abc", status="done", verdict="violation", violations=[
        Violation(type="Наличные оставлены без присмотра", timecode_sec=9.4, description="d",
                  evidence_frames=[EvidenceFrame(timestamp_sec=9.4, image_base64="xx")],
                  rule_id="R01", rule=RuleRef(act="Правила № 231", point="п. 20", strength="В"))])
    d = r.to_api()
    assert set(d) >= {"job_id", "status", "verdict", "violations"}
    v = d["violations"][0]
    assert set(v) >= {"type", "timecode_sec", "description", "evidence_frames"}
    assert v["evidence_frames"][0] == {"timestamp_sec": 9.4, "image_base64": "xx"}


def test_processing_result_has_no_verdict_or_violations():
    d = JobResult(job_id="abc", status="processing").to_api()
    assert d == {"job_id": "abc", "status": "processing"}


def test_no_violation_keeps_empty_list():
    d = JobResult(job_id="a", status="done", verdict="no_violation").to_api()
    assert d["violations"] == []


def _v(n_frames=1):
    return Violation(type="t", timecode_sec=1.5, description="d",
                     evidence_frames=[EvidenceFrame(timestamp_sec=1.5 + i, image_base64="xx") for i in range(n_frames)])


def test_done_without_verdict_still_carries_one():
    assert JobResult(job_id="a", status="done").to_api()["verdict"] == "no_violation"
    assert JobResult(job_id="a", status="done", violations=[_v()]).to_api()["verdict"] == "violation"


def test_evidence_frames_capped_at_three():
    d = JobResult(job_id="a", status="done", verdict="violation", violations=[_v(5)]).to_api()
    assert [f["timestamp_sec"] for f in d["violations"][0]["evidence_frames"]] == [1.5, 2.5, 3.5]


def test_hints_are_optional_and_only_on_done():
    assert "hints" not in JobResult(job_id="a", status="done", verdict="no_violation").to_api()
    assert "hints" not in JobResult(job_id="a", status="done", verdict="no_violation", hints=[]).to_api()
    d = JobResult(job_id="a", status="done", verdict="no_violation", hints=[_v()]).to_api()
    assert d["verdict"] == "no_violation" and d["violations"] == [] and d["hints"][0]["type"] == "t"
    for status in ("processing", "failed"):
        assert "hints" not in JobResult(job_id="a", status=status, hints=[_v()]).to_api()
