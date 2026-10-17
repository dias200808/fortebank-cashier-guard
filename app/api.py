import hmac
import logging
import subprocess
import threading
from contextlib import asynccontextmanager
from fractions import Fraction
from pathlib import Path

import av
import yaml
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import Settings
from app.jobs import JobStore, ProcessFn, Worker

APP_DIR = Path(__file__).parent
STATIC = APP_DIR / "static"
RULES_PATH, NORMS_PATH = APP_DIR / "rules.yaml", APP_DIR / "norms.yaml"
MB = 1 << 20
log = logging.getLogger(__name__)

PREVIEW = "preview.mp4"                                # next to input.mov in the job directory
PLAYABLE_CODECS = {"h264", "vp8", "vp9", "av1"}        # decoded by every current browser
PLAYABLE_FORMATS = {"mov", "mp4", "matroska", "webm"}  # FFmpeg demuxer names of the containers browsers read
PREVIEW_MAX_SIDE = 960                                 # the player needs no more; keeps the transcode cheap
PREVIEW_TB = Fraction(1, 1000)


def load_catalog(rules_path: Path = RULES_PATH, norms_path: Path = NORMS_PATH) -> dict:
    """Violation catalog: the enabled rules of rules.yaml (read at call time) + sources, principle and the
    not-checkable list from norms.yaml. Served at GET /api/v1/rules and rendered into README by scripts/catalog.py."""
    from app.pipeline.rules import load_rules  # lazy: the module pulls the tracker stack, stub/demo start without it
    norms = yaml.safe_load(Path(norms_path).read_text(encoding="utf-8"))
    rules = [{"id": r.id, "type": r.type,
              "summary": getattr(r, "summary", "") or "", "signals": getattr(r, "signals", "") or "",
              "strength": r.strength, "norm": {k: r.norm.get(k, "") for k in ("act", "point", "url", "quote")}}
             for r in sorted(load_rules(rules_path), key=lambda r: r.id)]
    # which rules cite each act, derived here so the list follows rules.yaml
    sources = [dict(s, rules=[{"id": r["id"], "point": r["norm"]["point"]} for r in rules if r["norm"]["url"] == s["url"]])
               for s in norms["sources"]]
    return {"rules": rules, "sources": sources, "principle": "\n\n".join(norms["principle"]),
            "strengths": norms["strengths"], "not_checkable": norms["not_checkable"], "checked": norms["checked"]}


def needs_preview(src: Path) -> bool:
    """True when browsers cannot play src as is: HEVC from iPhones, mpeg4, ProRes, 10-bit H.264, AVI..."""
    with av.open(str(src)) as c:
        if not c.streams.video:
            return False
        cc = c.streams.video[0].codec_context
        container_ok = bool(PLAYABLE_FORMATS & set(c.format.name.split(",")))
        h264_ok = cc.name != "h264" or cc.pix_fmt in ("yuv420p", "yuvj420p")
        return not (container_ok and cc.name in PLAYABLE_CODECS and h264_ok)


def _encode_pyav(src: Path, dst: Path) -> None:
    """Upright frames timed from stream start, as app.pipeline.frames decodes them, so timecodes and region boxes
    of the result match the player."""
    from app.pipeline.frames import apply_rotation
    with av.open(str(src)) as inp, av.open(str(dst), "w", format="mp4", options={"movflags": "+faststart"}) as out:
        vs = inp.streams.video[0]
        start, st, last = vs.start_time or 0, None, -1
        for frame in inp.decode(vs):
            if frame.pts is None:
                continue
            img = apply_rotation(frame.to_ndarray(format="rgb24"), getattr(frame, "rotation", 0) or 0)
            if st is None:
                h, w = img.shape[:2]
                k = min(1.0, PREVIEW_MAX_SIDE / max(w, h))
                st = out.add_stream("libx264", rate=30)
                st.width, st.height, st.pix_fmt = max(2, 2 * round(w * k / 2)), max(2, 2 * round(h * k / 2)), "yuv420p"
                st.time_base = st.codec_context.time_base = PREVIEW_TB
                st.thread_count = 2                  # leave the CPU to the analysis running at the same time
                st.options = {"preset": "veryfast", "crf": "23"}
            pts = max(0, round(float((frame.pts - start) * vs.time_base) / PREVIEW_TB))
            if pts <= last:
                continue
            last = pts
            f = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(width=st.width, height=st.height, format="yuv420p")
            f.pts, f.time_base = pts, PREVIEW_TB
            for packet in st.encode(f):
                out.mux(packet)
        if st is None:
            raise ValueError("no video frames")
        for packet in st.encode():
            out.mux(packet)


def _encode_ffmpeg(exe: str, src: Path, dst: Path) -> None:
    m = PREVIEW_MAX_SIDE
    scale = f"scale=w='if(gte(iw,ih),trunc(min({m},iw)/2)*2,-2)':h='if(gte(iw,ih),-2,trunc(min({m},ih)/2)*2)'"
    subprocess.run([exe, "-v", "error", "-nostdin", "-y", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2", "-vf", scale,
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(dst)],
                   check=True, capture_output=True, timeout=900)


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg  # optional, not in requirements.txt
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _has_libx264() -> bool:
    try:
        av.codec.Codec("libx264", "w")
        return True
    except Exception:
        return False


def make_preview(src: Path, dst: Path) -> bool:
    """Write a browser-playable H.264 .mp4 copy of src to dst when src itself does not play in browsers.
    Encoder: PyAV's libx264, else the ffmpeg of imageio-ffmpeg; with neither the original stays the only file
    (the UI then says the video does not play). True when dst was written."""
    if dst.exists() or not needs_preview(src):
        return False
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    try:
        if _has_libx264():
            _encode_pyav(src, tmp)
        elif exe := _ffmpeg_exe():
            _encode_ffmpeg(exe, src, tmp)
        else:
            log.warning("%s: not browser-playable and no H.264 encoder available, serving the original", src)
            return False
        tmp.replace(dst)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def create_app(settings: Settings, process: ProcessFn) -> FastAPI:
    store = JobStore(settings.jobs_dir)
    worker = Worker(store, process)
    preview_lock = threading.Lock()  # one transcode at a time

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        yield

    app = FastAPI(title="Cash discipline video checker", lifespan=lifespan)

    def key_ok(key: str | None) -> bool:
        return bool(key) and hmac.compare_digest(key.encode(), settings.api_key.encode())

    def check_key(*keys: str | None) -> None:
        """401 unless one of the supplied credentials is the API key (constant-time compare)."""
        if not any([key_ok(k) for k in keys]):  # list: compare every candidate, no short-circuit
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    def preview_job(job_id: str) -> None:
        """Background, off the analysis path: a failure here only means the player gets the original file."""
        src = store.video_path(job_id)
        try:
            with preview_lock:
                if make_preview(src, src.with_name(PREVIEW)):
                    log.info("job %s: H.264 preview written for the player", job_id)
        except Exception as e:
            log.info("job %s: no browser preview (%s: %s)", job_id, type(e).__name__, e)

    @app.post("/api/v1/analyze")
    async def analyze(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        job_id, path = store.new_job()
        size = 0
        try:
            with open(path, "wb") as f:
                while chunk := await file.read(MB):
                    size += len(chunk)
                    if size > settings.max_upload_mb * MB:
                        raise HTTPException(413, "file too large")
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, "empty file")
        except HTTPException as e:
            store.fail(job_id, str(e.detail))
            raise
        worker.submit(job_id)
        threading.Thread(target=preview_job, args=(job_id,), name=f"preview-{job_id}", daemon=True).start()
        return {"job_id": job_id, "status": "processing"}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, x_api_key: str | None = Header(default=None)):
        check_key(x_api_key)
        r = store.load(job_id)
        if r is None:
            raise HTTPException(404, "job not found")
        return JSONResponse(r.to_api())

    @app.get("/api/v1/jobs/{job_id}/video")
    def get_job_video(job_id: str, key: str | None = Query(default=None),
                      x_api_key: str | None = Header(default=None)):
        """The uploaded video, or its H.264 preview when the original codec does not play in browsers, with Range
        support so a <video> can seek. A <video src> cannot send headers, so the key may also come as ?key=."""
        check_key(x_api_key, key)
        path = store.video_path(job_id)
        if store.load(job_id) is None or not path.is_file():
            raise HTTPException(404, "job not found")
        preview = path.with_name(PREVIEW)
        if preview.is_file():
            return FileResponse(preview, media_type="video/mp4")
        return FileResponse(path, media_type="video/quicktime")  # Starlette serves Range -> 206

    @app.get("/api/v1/rules")
    def get_rules(key: str | None = Query(default=None), x_api_key: str | None = Header(default=None)):
        """Catalog of violation types with norm quotes, the sources and the matching principle."""
        check_key(x_api_key, key)
        return load_catalog()

    @app.get("/health")
    def health():
        return {"ok": True, "queued": worker.q.qsize()}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
