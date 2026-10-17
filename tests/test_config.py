from pathlib import Path

from app.config import Settings, load_settings


def test_defaults():
    s = Settings()
    assert s.job_limit_s == 270.0 and s.det_batch == 8 and s.unknown_channel is True


def test_load_settings_reads_env(monkeypatch):
    env = {"VLM_WORKERS": "4", "DET_FPS": "2.5", "PROBE_FPS": "2", "MAX_PROBE_FRAMES": "100", "MAX_UPLOAD_MB": "50",
           "JOB_LIMIT_S": "240", "DET_BATCH": "4", "UNKNOWN_CHANNEL": "false", "JOBS_DIR": "/tmp/j"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = load_settings()
    assert (s.vlm_workers, s.det_fps, s.probe_fps, s.max_probe_frames, s.max_upload_mb) == (4, 2.5, 2.0, 100, 50)
    assert (s.job_limit_s, s.det_batch, s.unknown_channel, s.jobs_dir) == (240.0, 4, False, Path("/tmp/j"))


def test_unknown_channel_env_values(monkeypatch):
    for value, expected in (("0", False), ("FALSE", False), ("1", True), ("true", True)):
        monkeypatch.setenv("UNKNOWN_CHANNEL", value)
        assert load_settings().unknown_channel is expected
    monkeypatch.delenv("UNKNOWN_CHANNEL")
    assert load_settings().unknown_channel is True
