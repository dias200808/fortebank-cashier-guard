import time
from pathlib import Path

import av
import numpy as np
import pytest
import yaml

import app.api as api
from app.api import load_catalog, make_preview, needs_preview
from app.pipeline.rules import load_rules
from scripts.catalog import render
from tests.conftest import write_video
from tests.test_api import KEY, make_client, upload

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules(ROOT / "app" / "rules.yaml")


def test_rules_requires_key(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/api/v1/rules").status_code == 401
        assert c.get("/api/v1/rules", headers={"X-API-Key": "bad"}).status_code == 401
        assert c.get("/api/v1/rules", params={"key": "bad"}).status_code == 401


def test_rules_with_header_or_query_key(tmp_path):
    with make_client(tmp_path) as c:
        a = c.get("/api/v1/rules", headers=KEY)
        b = c.get("/api/v1/rules", params={"key": "k"})
        assert a.status_code == b.status_code == 200 and a.json() == b.json()


def test_rules_shape_follows_rules_yaml(tmp_path):
    with make_client(tmp_path) as c:
        d = c.get("/api/v1/rules", headers=KEY).json()
    assert [r["id"] for r in d["rules"]] == sorted(r.id for r in RULES)
    by_id = {r.id: r for r in RULES}
    for r in d["rules"]:
        assert set(r) == {"id", "type", "summary", "signals", "strength", "norm"}
        src = by_id[r["id"]]
        assert r["type"] == src.type and r["strength"] == src.strength in {"В", "С", "Н"}
        assert r["norm"] == {k: src.norm[k] for k in ("act", "point", "url", "quote")}
        assert r["summary"] == (getattr(src, "summary", "") or "") and isinstance(r["signals"], str)
    assert d["principle"].strip() and "\n\n" in d["principle"]
    assert {s["code"] for s in d["sources"]} >= {"A", "B", "G"}
    for s in d["sources"]:
        assert s["act"] and s["used_for"] and s["url"].startswith("https://adilet.zan.kz/")
        assert s["rules"] == [{"id": r["id"], "point": r["norm"]["point"]} for r in d["rules"] if r["norm"]["url"] == s["url"]]
    assert sum(len(s["rules"]) for s in d["sources"]) == len(d["rules"])      # every rule cites a listed act
    assert set(d["strengths"]) == {"В", "С", "Н"}
    assert all(x["item"] and x["why"] for x in d["not_checkable"])


def test_catalog_reads_rules_at_call_time(tmp_path):
    rules = yaml.safe_load((ROOT / "app" / "rules.yaml").read_text(encoding="utf-8"))
    first = dict(rules[0], id="R99")
    off = dict(rules[1], id="R98", enabled=False)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump([first, off], allow_unicode=True), encoding="utf-8")
    cat = load_catalog(p)
    assert [r["id"] for r in cat["rules"]] == ["R99"]          # disabled rules are not in the catalog
    assert cat["rules"][0]["type"] == rules[0]["type"]


def test_readme_markdown_lists_every_rule():
    md = render(load_catalog())
    for r in RULES:
        assert f"| {r.id} |" in md and r.type in md
    assert "## 5. Типы нарушений" in md and "## 6. Нормативные источники и принцип сопоставления" in md
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count("<!-- catalog:start -->") == 1 and readme.count("<!-- catalog:end -->") == 1


def test_index_links_the_catalog(tmp_path):
    with make_client(tmp_path) as c:
        body = c.get("/").text
    assert "Нарушения и источники" in body and "/api/v1/rules" in body and "#rules" in body


# ---------- browser preview for codecs browsers cannot play ----------

def ramp(i: int) -> np.ndarray:
    img = np.zeros((160, 96, 3), np.uint8)
    img[:, : (i * 4) % 96] = 220
    return img


def h264_video(path: Path, n: int = 20, fps: int = 10) -> Path:
    c = av.open(str(path), "w")
    s = c.add_stream("libx264", rate=fps)
    s.width, s.height, s.pix_fmt = 96, 160, "yuv420p"
    for i in range(n):
        for p in s.encode(av.VideoFrame.from_ndarray(ramp(i), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return path


def probe(path: Path) -> tuple[str, int, float]:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n = sum(1 for _ in c.decode(s))
        return s.codec_context.name, n, float(s.duration * s.time_base)


def test_needs_preview_only_for_non_browser_codecs(tmp_path):
    assert needs_preview(write_video(tmp_path / "a.mov", 10, 10, ramp)) is True     # mpeg4
    assert needs_preview(h264_video(tmp_path / "b.mov")) is False


def test_preview_is_h264_with_same_frames_and_duration(tmp_path):
    src = write_video(tmp_path / "in.mov", 30, 10, ramp)
    assert make_preview(src, tmp_path / "p.mp4") is True
    codec, n, dur = probe(tmp_path / "p.mp4")
    assert codec == "h264" and n == 30 and abs(dur - 3.0) < 0.15
    assert not (tmp_path / "p.part.mp4").exists()
    assert make_preview(src, tmp_path / "p.mp4") is False                            # already there


def test_preview_of_hevc(tmp_path):
    try:
        av.codec.Codec("libx265", "w")
    except Exception:
        pytest.skip("no HEVC encoder in this PyAV build")
    c = av.open(str(tmp_path / "hevc.mov"), "w")
    s = c.add_stream("libx265", rate=10, options={"log-level": "error"})
    s.width, s.height, s.pix_fmt = 96, 160, "yuv420p"
    for i in range(20):
        for p in s.encode(av.VideoFrame.from_ndarray(ramp(i), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    assert make_preview(tmp_path / "hevc.mov", tmp_path / "p.mp4") is True
    assert probe(tmp_path / "p.mp4")[:2] == ("h264", 20)


def test_preview_falls_back_to_ffmpeg_then_gives_up(tmp_path, monkeypatch):
    src = write_video(tmp_path / "in.mov", 20, 10, ramp)
    monkeypatch.setattr(api, "_has_libx264", lambda: False)
    if api._ffmpeg_exe():
        assert make_preview(src, tmp_path / "ff.mp4") is True
        assert probe(tmp_path / "ff.mp4")[:2] == ("h264", 20)
    monkeypatch.setattr(api, "_ffmpeg_exe", lambda: None)
    assert make_preview(src, tmp_path / "none.mp4") is False and not (tmp_path / "none.mp4").exists()


def wait_for(path: Path, timeout: float = 20.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def test_video_endpoint_serves_preview_with_range(tmp_path):
    data = write_video(tmp_path / "src.mov", 30, 10, ramp).read_bytes()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    with make_client(jobs) as c:
        jid = upload(c, data)
        preview = jobs / jid / api.PREVIEW
        assert wait_for(preview)
        r = c.get(f"/api/v1/jobs/{jid}/video", params={"key": "k"})
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
        assert r.content == preview.read_bytes()
        r = c.get(f"/api/v1/jobs/{jid}/video", headers={**KEY, "Range": "bytes=0-99"})
        assert r.status_code == 206 and r.headers["content-length"] == "100"
        assert (jobs / jid / "input.mov").read_bytes() == data                     # the original stays for analysis


def test_browser_playable_upload_gets_no_preview(tmp_path):
    data = h264_video(tmp_path / "src.mov").read_bytes()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    with make_client(jobs) as c:
        jid = upload(c, data)
        assert not wait_for(jobs / jid / api.PREVIEW, timeout=1.0)
        r = c.get(f"/api/v1/jobs/{jid}/video", headers=KEY)
        assert r.headers["content-type"].startswith("video/quicktime") and r.content == data
