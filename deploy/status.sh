#!/usr/bin/env bash
# Show what is running, queue size, public URL, GPU memory and the last log lines.
cd "$(dirname "$0")/.."
[ -f deploy/.env ] && { set -a; . deploy/.env; set +a; }
for p in vllm app tunnel; do
  if [ -f "deploy/run/$p.pid" ] && kill -0 "$(cat "deploy/run/$p.pid")" 2>/dev/null; then echo "$p: работает"; else echo "$p: НЕ работает"; fi
done
echo "health: $(curl -s -m 5 "http://127.0.0.1:${PORT:-8080}/health" || echo 'нет ответа')"
echo "публичный адрес: $(cat deploy/run/public_url 2>/dev/null || echo '—')"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- app.log (последние строки)"; tail -n 8 deploy/logs/app.log 2>/dev/null
echo "--- vllm.log (последние строки)"; tail -n 4 deploy/logs/vllm.log 2>/dev/null
