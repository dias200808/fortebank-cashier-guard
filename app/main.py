import os

from app.api import create_app
from app.config import load_settings

settings = load_settings()
mode = os.environ.get("PIPELINE", "real")
if mode == "stub":
    from app.stub import stub_process as process
elif mode == "demo":  # UI showcase with labelled example violations; never an analysis result
    from app.demo import demo_process as process
else:
    if settings.api_key == "dev-key":
        raise RuntimeError("set API_KEY (the default dev-key is only allowed with PIPELINE=stub or PIPELINE=demo)")
    from app.pipeline.run import build_processor
    process = build_processor(settings)
app = create_app(settings, process)
