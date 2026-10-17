from pathlib import Path

import numpy as np

from app.pipeline.frames import VideoFrames
from app.pipeline.probes import Fact, load_facts, probe_times, run_probes
from app.pipeline.scene import Zones
from app.pipeline.vlm import FakeVLM


def test_facts_yaml_loads():
    facts = load_facts(Path("app/facts.yaml"))
    ids = {f.id for f in facts}
    assert {"F_cash_desk", "F_safe_open", "F_door_open", "F_screen_on", "F_handover"} <= ids
    assert all(f.zone in {"desk", "safe", "door", "laptop", "full"} for f in facts)


def test_probe_times_base_plus_extra_and_cap():
    assert probe_times(3.0, 1.0, [1.4, 1.4, 9.0], cap=100) == [0.0, 1.0, 1.4, 2.0, 3.0]
    ts = probe_times(10.0, 1.0, [i / 10 for i in range(100)], cap=15)
    assert len(ts) == 15 and ts[0] == 0.0 and 10.0 in ts


def test_probe_times_long_video_subsamples_base():
    assert len(probe_times(500.0, 1.0, [], cap=100)) <= 100


def test_run_probes_series_follow_frames():
    imgs = [np.full((100, 60, 3), 255 if i >= 5 else 0, np.uint8) for i in range(10)]
    vf = VideoFrames([float(i) for i in range(10)], imgs, 9.0)
    facts = [Fact("F_cash_desk", "desk", "banknotes?"), Fact("F_door_open", "door", "door?")]
    vlm = FakeVLM(answer=lambda ims, q: 0.9 if ("banknotes" in q and ims[0].mean() > 100) else 0.1)
    s = run_probes(vf, Zones(desk=(0, 0.5, 1, 1)), vlm, facts, [0.0, 4.0, 5.0, 9.0])
    assert [p for _, p in s["F_cash_desk"]] == [0.1, 0.1, 0.9, 0.9]
    assert [p for _, p in s["F_door_open"]] == [0.1] * 4
