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
