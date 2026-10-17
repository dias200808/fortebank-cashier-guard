import os

from app.api import create_app
from app.config import load_settings
from app.stub import stub_process

settings = load_settings()
if os.environ.get("PIPELINE", "real") == "stub":
    process = stub_process
else:
    if settings.api_key == "dev-key":
        raise RuntimeError("set API_KEY (the default dev-key is only allowed with PIPELINE=stub)")
    from app.pipeline.run import build_processor
    process = build_processor(settings)
app = create_app(settings, process)
