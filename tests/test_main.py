import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_main(tmp_path, **env):
    e = {k: v for k, v in os.environ.items() if k not in {"API_KEY", "PIPELINE"}}
    e.update(PYTHONPATH=str(ROOT), JOBS_DIR=str(tmp_path / "jobs"), VLM_BACKEND="fake", **env)
    return subprocess.run([sys.executable, "-c", "import app.main"], cwd=tmp_path, env=e, capture_output=True, text=True)


def test_real_pipeline_refuses_default_api_key(tmp_path):
    r = run_main(tmp_path, PIPELINE="real")
    assert r.returncode != 0 and "set API_KEY" in r.stderr


def test_stub_pipeline_allows_default_api_key(tmp_path):
    assert run_main(tmp_path, PIPELINE="stub").returncode == 0
