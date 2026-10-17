#!/usr/bin/env bash
# Bring back the VLM that was served before the last switch_vlm.sh (tunnel and public URL stay).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f deploy/.env.prev-vlm ] || { echo "Нет deploy/.env.prev-vlm — откатывать нечего"; exit 1; }
cp deploy/.env.prev-vlm deploy/.env
grep -E "^VLM_(MODEL|GPU_UTIL|MAX_LEN|EXTRA_ARGS)=" deploy/.env
bash deploy/stop.sh keep-tunnel
for _ in $(seq 1 60); do pgrep -f "vllm serve" >/dev/null || break; sleep 1; done
bash deploy/start.sh
