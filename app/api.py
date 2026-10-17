import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
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

    def check_key(key: str | None) -> None:
        if not key or not hmac.compare_digest(key.encode(), settings.api_key.encode()):
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

    @app.get("/health")
    def health():
        return {"ok": True, "queued": worker.q.qsize()}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
