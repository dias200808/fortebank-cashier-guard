#!/usr/bin/env bash
# A/B: switch the served VLM (weights must already be downloaded). The previous deploy/.env is saved to
# deploy/.env.prev-vlm, so `bash deploy/revert_vlm.sh` brings the old model back in ~2–3 minutes.
# The cloudflared tunnel keeps running: the public URL does not change.
# Usage: bash deploy/switch_vlm.sh RedHatAI/Qwen3.5-9B-FP8-dynamic "--no-enable-prefix-caching" [gpu_util] [max_len]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; EXTRA="${2:-}"; UTIL="${3:-}"; MAXLEN="${4:-}"
cp deploy/.env deploy/.env.prev-vlm
set_kv() { if grep -q "^$1=" deploy/.env; then sed -i "s|^$1=.*|$1=$2|" deploy/.env; else echo "$1=$2" >> deploy/.env; fi; }
set_kv VLM_MODEL "$MODEL"
set_kv VLM_EXTRA_ARGS "\"$EXTRA\""
[ -n "$UTIL" ] && set_kv VLM_GPU_UTIL "$UTIL"
[ -n "$MAXLEN" ] && set_kv VLM_MAX_LEN "$MAXLEN"
echo "=== было:"; grep -E "^VLM_(MODEL|GPU_UTIL|MAX_LEN|EXTRA_ARGS)=" deploy/.env.prev-vlm || true
echo "=== стало:"; grep -E "^VLM_(MODEL|GPU_UTIL|MAX_LEN|EXTRA_ARGS)=" deploy/.env
bash deploy/stop.sh keep-tunnel
bash deploy/start.sh || { echo "Старт не удался — откат"; bash deploy/revert_vlm.sh; exit 1; }
bash deploy/check_vlm.sh
