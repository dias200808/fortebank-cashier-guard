#!/usr/bin/env bash
# The answer must start with Yes/No (no thinking) and carry top_logprobs — our P(yes) reads exactly that token.
cd "$(dirname "$0")/.."
set -a; . deploy/.env; set +a
.venv/bin/python - <<'PY'
import os, numpy as np
from app.pipeline.vlm import OpenAIVLM
v = OpenAIVLM(f"http://127.0.0.1:{os.environ.get('VLM_PORT', '18000')}/v1", os.environ["VLM_MODEL"], workers=4)
img = np.full((480, 480, 3), 255, np.uint8); img[140:340, 140:340] = (220, 30, 30)
p_red, p_blue = v.yes_probs([([img], "Is there a red square in the image? Answer Yes or No."), ([img], "Is there a blue circle in the image? Answer Yes or No.")])
print(f"P(yes) red square = {p_red:.3f} (ожидаем > 0.9), blue circle = {p_blue:.3f} (ожидаем < 0.1)")
print("OK" if p_red > 0.9 and p_blue < 0.1 else "ПРОВАЛ: вероятности да/нет не работают")
PY
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
