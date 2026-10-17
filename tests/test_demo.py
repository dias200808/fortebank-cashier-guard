import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from app.demo import demo_process

ROOT = Path(__file__).resolve().parents[1]
RULES = {r["id"]: r for r in yaml.safe_load((ROOT / "app" / "rules.yaml").read_text(encoding="utf-8"))}


def gradient(i: int) -> np.ndarray:
    img = np.zeros((96, 160, 3), np.uint8)
    img[:, :, 0] = (i * 3) % 256
    img[:, : 10 + i % 140, 1] = 200
    return img


def test_demo_process_returns_two_labelled_violations(tmp_video):
    path = tmp_video(100, 10, gradient)  # 10 s
    r = demo_process(path, "abc123", delay_s=0)
    assert r.job_id == "abc123" and r.status == "done" and r.verdict == "violation"
    assert len(r.violations) == 2
    for v in r.violations:
        assert v.type.startswith("ДЕМО — ")
        assert v.description.startswith("[Демо-режим: это не результат анализа] ")
        assert 1 <= len(v.evidence_frames) <= 3
        assert v.confidence == 0.0
        for f in v.evidence_frames:
            assert f.image_base64.startswith("/9j/")  # JPEG magic, base64-encoded
            assert 0.0 <= f.timestamp_sec <= 10.0


def test_demo_process_timecodes_and_rules(tmp_video):
    path = tmp_video(100, 10, gradient)
    v1, v2 = demo_process(path, "j", delay_s=0).violations
    assert abs(v1.timecode_sec - 2.5) <= 0.5 and abs(v2.timecode_sec - 6.0) <= 0.5
    for v, rid in ((v1, "R01"), (v2, "R08")):
        norm = RULES[rid]["norm"]
        assert v.rule_id == rid
        assert v.rule.act == norm["act"] and v.rule.point == norm["point"] and v.rule.url == norm["url"]
        assert v.rule.strength == RULES[rid]["strength"]
        assert v.type == "ДЕМО — " + RULES[rid]["type"]


def test_demo_process_frames_come_from_the_video(tmp_video):
    path = tmp_video(100, 10, gradient)
    v1, v2 = demo_process(path, "j", delay_s=0).violations
    assert v1.evidence_frames[0].image_base64 != v2.evidence_frames[0].image_base64
    times = [f.timestamp_sec for f in v1.evidence_frames]
    assert times == sorted(set(times))


def test_demo_process_short_video(tmp_video):
    path = tmp_video(5, 10, gradient)  # 0.5 s
    r = demo_process(path, "j", delay_s=0)
    assert len(r.violations) == 2 and all(1 <= len(v.evidence_frames) <= 3 for v in r.violations)


def test_demo_api_result_shape(tmp_video):
    d = demo_process(tmp_video(30, 10, gradient), "j", delay_s=0).to_api()
    assert d["verdict"] == "violation" and len(d["violations"]) == 2
    assert set(d["violations"][0]["rule"]) == {"act", "point", "strength", "url"}


def run_main(tmp_path, **env):
    e = {k: v for k, v in os.environ.items() if k not in {"API_KEY", "PIPELINE"}}
    e.update(PYTHONPATH=str(ROOT), JOBS_DIR=str(tmp_path / "jobs"), **env)
    code = "import sys, app.main; print('app.pipeline.run' in sys.modules, app.main.process.__module__)"
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=e, capture_output=True, text=True)


def test_main_demo_pipeline_allows_default_key_and_skips_real_pipeline(tmp_path):
    r = run_main(tmp_path, PIPELINE="demo")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "app.demo"]
