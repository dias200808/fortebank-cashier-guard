import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import Settings
from app.jobs import JobStore, ProcessFn, Worker

STATIC = Path(__file__).parent / "static"
MB = 1 << 20


def create_app(settings: Settings, process: ProcessFn) -> FastAPI:
    store = JobStore(settings.jobs_dir)
    worker = Worker(store, process)

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
        """The uploaded video, with Range support so a <video> can seek. A <video src> cannot send
        headers, so the key may also come as ?key=."""
        check_key(x_api_key, key)
        path = store.video_path(job_id)
        if store.load(job_id) is None or not path.is_file():
            raise HTTPException(404, "job not found")
        return FileResponse(path, media_type="video/quicktime")  # Starlette serves Range -> 206

    @app.get("/health")
    def health():
        return {"ok": True, "queued": worker.q.qsize()}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
