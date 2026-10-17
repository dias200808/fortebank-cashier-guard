#!/usr/bin/env bash
# Stop the service, the model server and the tunnel. "keep-tunnel" keeps the public URL alive.
cd "$(dirname "$0")/.."
for p in app vllm tunnel; do
  [ "$p" = tunnel ] && [ "${1:-}" = keep-tunnel ] && { echo "туннель оставлен"; continue; }
  if [ -f "deploy/run/$p.pid" ]; then
    kill "$(cat "deploy/run/$p.pid")" 2>/dev/null && echo "остановлен: $p"
    rm -f "deploy/run/$p.pid"
  fi
done
