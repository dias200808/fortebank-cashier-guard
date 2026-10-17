#!/usr/bin/env bash
# Start the model server (vLLM), the service (FastAPI) and a public HTTPS tunnel.
# Run after deploy/setup.sh. Safe to re-run: already running parts are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE="deploy/.env"
[ -f "$ENV_FILE" ] || { echo "Нет $ENV_FILE — сначала bash deploy/setup.sh"; exit 1; }
set -a; . "$ENV_FILE"; set +a
mkdir -p deploy/logs deploy/run
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

say() { printf '\n=== %s\n' "$*"; }
alive() { [ -f "deploy/run/$1.pid" ] && kill -0 "$(cat "deploy/run/$1.pid")" 2>/dev/null; }
wait_url() {  # url seconds label [pid name: give up as soon as that process has died]
  for _ in $(seq 1 "$2"); do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    if [ -n "${4:-}" ] && ! alive "$4"; then echo "Процесс $4 завершился: $3. Смотрите deploy/logs/$4.log"; return 1; fi
    sleep 1
  done
  echo "Не дождался: $3 ($1). Смотрите deploy/logs/"; return 1
}

say "Модель: $VLM_MODEL"
if alive vllm; then echo "vLLM уже запущен"; else
  nohup .venv-vllm/bin/vllm serve "$VLM_MODEL" --host 127.0.0.1 --port "${VLM_PORT:-18000}" \
    --max-model-len "$VLM_MAX_LEN" --gpu-memory-utilization "$VLM_GPU_UTIL" \
    --limit-mm-per-prompt '{"image": 10}' --max-num-seqs 16 \
    > deploy/logs/vllm.log 2>&1 &
  echo $! > deploy/run/vllm.pid
  echo "Загрузка модели (первый раз до ~10 минут)…"
fi
wait_url http://127.0.0.1:${VLM_PORT:-18000}/v1/models 900 "vLLM" vllm

say "Проверка logprobs (нужны для вероятности «да»)"
LP=$(curl -s "http://127.0.0.1:${VLM_PORT:-18000}/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\": \"$VLM_MODEL\", \"max_tokens\": 1, \"temperature\": 0, \"logprobs\": true, \"top_logprobs\": 5, \"messages\": [{\"role\": \"user\", \"content\": \"Is the sky blue? Answer with exactly one word: Yes or No.\"}]}")
if echo "$LP" | grep -q top_logprobs; then echo "OK"; else echo "ВНИМАНИЕ: logprobs нет в ответе: $LP"; fi

say "Формат координат модели (rel1000 или abs)"
MODE=$(.venv/bin/python - "$VLM_MODEL" "http://127.0.0.1:${VLM_PORT:-18000}/v1" <<'PY' 2>/dev/null || echo rel1000
import sys
import numpy as np
from app.pipeline.scene import parse_bbox
from app.pipeline.vlm import OpenAIVLM
img = np.full((848, 464, 3), 200, np.uint8)
img[424:636, 232:406] = (220, 30, 30)
truth = (0.5, 0.5, 0.875, 0.75)
text = OpenAIVLM(sys.argv[2], sys.argv[1]).text(
    [img], 'Locate the red rectangle. Reply with JSON only, like {"bbox_2d": [x1, y1, x2, y2]}.', 64)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
scores = {m: (iou(parse_bbox(text, 464, 848, m), truth) if parse_bbox(text, 464, 848, m) else 0.0) for m in ("rel1000", "abs")}
print(max(scores, key=scores.get) if max(scores.values()) > 0.5 else "rel1000")
PY
)
echo "VLM_COORD_MODE=$MODE"
sed -i.bak "s/^VLM_COORD_MODE=.*/VLM_COORD_MODE=$MODE/" "$ENV_FILE" && rm -f "$ENV_FILE.bak"
export VLM_COORD_MODE="$MODE"

say "Сервис (порт $PORT)"
if alive app; then echo "Сервис уже запущен"; else
  VLM_URL="http://127.0.0.1:${VLM_PORT:-18000}/v1" nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 \
    > deploy/logs/app.log 2>&1 &
  echo $! > deploy/run/app.pid
fi
wait_url "http://127.0.0.1:$PORT/health" 180 "сервис" app

say "Публичный адрес (Cloudflare quick tunnel)"
if alive tunnel; then echo "Туннель уже запущен"; else
  nohup deploy/bin/cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" > deploy/logs/tunnel.log 2>&1 &
  echo $! > deploy/run/tunnel.pid
fi
URL=""
for _ in $(seq 1 30); do URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' deploy/logs/tunnel.log | head -1 || true); [ -n "$URL" ] && break; sleep 1; done
[ -n "$URL" ] && echo "$URL" > deploy/run/public_url

say "ГОТОВО"
echo "Публичный адрес: ${URL:-не получен — смотрите deploy/logs/tunnel.log}"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -n "$IP" ] && echo "Адрес в сети:     http://$IP:$PORT"
echo "API-ключ:         $API_KEY"
echo "Сайт:             ${URL:-http://$IP:$PORT}/?key=$API_KEY"
echo
echo "Проверка:  bash deploy/smoke.sh"
echo "Статус:    bash deploy/status.sh"
echo "Остановка: bash deploy/stop.sh   (туннель не перезапускайте после сдачи — адрес сменится)"
