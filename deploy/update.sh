#!/usr/bin/env bash
# Update the code from a new archive and restart only the service: the model (vLLM) and the
# public tunnel keep running, so the public URL stays the same.
# Usage: bash deploy/update.sh /path/to/fortebank-cashier-guard.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
ARCHIVE="${1:-}"
[ -f "$ARCHIVE" ] || { echo "Укажите архив: bash deploy/update.sh /fortebank-cashier-guard.tar.gz"; exit 1; }
set -a; . deploy/.env; set +a

REQ_BEFORE=$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1 || true)
tar -xzf "$ARCHIVE" -C .          # deploy/.env, venvs, logs, jobs are not in the archive and stay as they are
REQ_AFTER=$(md5sum requirements.txt | cut -d' ' -f1)
if [ "$REQ_BEFORE" != "$REQ_AFTER" ]; then echo "requirements.txt изменился — ставлю зависимости"; .venv/bin/pip install -q -r requirements.txt; fi

if [ -f deploy/run/app.pid ]; then kill "$(cat deploy/run/app.pid)" 2>/dev/null || true; rm -f deploy/run/app.pid; sleep 2; fi
VLM_URL="http://127.0.0.1:${VLM_PORT:-18000}/v1" nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 \
  >> deploy/logs/app.log 2>&1 &
echo $! > deploy/run/app.pid
for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && echo "Обновлено, сервис работает. Адрес прежний: $(cat deploy/run/public_url 2>/dev/null || echo '—')" \
  || { echo "Сервис не поднялся — смотрите tail -50 deploy/logs/app.log"; exit 1; }
