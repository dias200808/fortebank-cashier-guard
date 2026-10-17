#!/usr/bin/env bash
# One-time setup on a Linux GPU host (Ubuntu 22.04/24.04 with an NVIDIA driver).
# Creates two virtualenvs (vLLM model server, app), downloads model weights once,
# fetches cloudflared and writes deploy/.env with a fresh API key.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
ENV_FILE="deploy/.env"
mkdir -p deploy/bin deploy/logs deploy/run

say() { printf '\n=== %s\n' "$*"; }

command -v nvidia-smi >/dev/null || { echo "Нет nvidia-smi: установите драйвер NVIDIA"; exit 1; }
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
say "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${VRAM_MB} MiB"

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Нужен Python 3.10+"; exit 1; }
if ! "$PY" -m venv --help >/dev/null 2>&1 && command -v apt-get >/dev/null; then
  say "Ставлю python3-venv"; (sudo -n true 2>/dev/null && sudo apt-get install -y python3-venv) || apt-get install -y python3-venv
fi

# Model choice by VRAM (override with VLM_MODEL=... bash deploy/setup.sh)
if [ -z "${VLM_MODEL:-}" ]; then
  if   [ "$VRAM_MB" -ge 22000 ]; then VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct-FP8"; FALLBACK="Qwen/Qwen3-VL-8B-Instruct"
  elif [ "$VRAM_MB" -ge 11000 ]; then VLM_MODEL="Qwen/Qwen3-VL-4B-Instruct";     FALLBACK=""
  else                                VLM_MODEL="Qwen/Qwen3-VL-2B-Instruct";     FALLBACK=""
  fi
else
  FALLBACK=""
fi

say "venv для vLLM (.venv-vllm)"
[ -d .venv-vllm ] || "$PY" -m venv .venv-vllm
.venv-vllm/bin/pip install -q -U pip
.venv-vllm/bin/pip install -q "vllm>=0.11" huggingface_hub

say "venv для сервиса (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt

say "Скачиваю веса модели: $VLM_MODEL"
dl() { .venv-vllm/bin/python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$1"; }
if ! dl "$VLM_MODEL"; then
  if [ -n "$FALLBACK" ]; then
    echo "Не удалось скачать $VLM_MODEL, беру $FALLBACK"; VLM_MODEL="$FALLBACK"; dl "$VLM_MODEL"
  else
    echo "Не удалось скачать $VLM_MODEL"; exit 1
  fi
fi

say "Веса детектора людей (torchvision Faster R-CNN R50-FPN-v2)"
.venv/bin/python -c "import torchvision.models.detection as d; d.fasterrcnn_resnet50_fpn_v2(weights=d.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)"

say "Веса детектора предметов (OWLv2, Apache-2.0)"
OWL_OK=1
.venv/bin/python -c "import sys, transformers; from huggingface_hub import snapshot_download; assert transformers.is_torch_available(), 'torch < 2.5'; snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.txt', 'model.safetensors'])" \
  google/owlv2-base-patch16-ensemble || { OWL_OK=0; echo "OWLv2 недоступен (веса или torch < 2.5) — сервис будет работать без него"; }

say "cloudflared (публичный HTTPS-адрес через туннель)"
if [ ! -x deploy/bin/cloudflared ]; then
  ARCH=$(uname -m); case "$ARCH" in x86_64) CF=amd64;; aarch64|arm64) CF=arm64;; *) CF=amd64;; esac
  curl -fsSL -o deploy/bin/cloudflared "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$CF"
  chmod +x deploy/bin/cloudflared
fi

# Memory budget: FP8/4B/2B fit comfortably; bf16 8B on 24 GB needs a larger share.
# FP8 8B on 24 GB: 0.72 (~17.3 GiB: ~10 GiB weights, ~5 GiB KV = 2+ requests of 16k tokens) leaves room for the
# service (~4.5 GB) plus OWLv2 (~1-2 GB). Other models, or no OWLv2, keep their share (same as enable_owl.sh).
GPU_UTIL=0.80; MAX_LEN=16384; OBJ=off
case "$VLM_MODEL" in
  *8B-Instruct) GPU_UTIL=0.86; MAX_LEN=8192;;
  *8B-Instruct-FP8) if [ "$OWL_OK" = 1 ]; then GPU_UTIL=0.72; OBJ=owlv2; fi;;
esac

if [ -f "$ENV_FILE" ] && grep -q '^API_KEY=' "$ENV_FILE"; then
  API_KEY=$(grep '^API_KEY=' "$ENV_FILE" | cut -d= -f2-)
else
  API_KEY=$(.venv/bin/python -c "import secrets; print(secrets.token_hex(24))")
fi
# Vast.ai/RunPod images often run Jupyter on 8080 inside the container: pick a free port.
port_free() { .venv/bin/python -c "import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', int(sys.argv[1])))" "$1" 2>/dev/null; }
APP_PORT=8080; port_free 8080 || APP_PORT=18080
VLM_PORT=18000; port_free 18000 || VLM_PORT=18001
cat > "$ENV_FILE" <<EOF
VLM_MODEL=$VLM_MODEL
VLM_GPU_UTIL=$GPU_UTIL
VLM_MAX_LEN=$MAX_LEN
VLM_COORD_MODE=rel1000
API_KEY=$API_KEY
PORT=$APP_PORT
VLM_PORT=$VLM_PORT
DETECTOR=frcnn_r50v2
DEVICE=cuda
VLM_WORKERS=16
OBJ_DETECTOR=$OBJ
EOF
chmod 600 "$ENV_FILE"

say "Готово. Дальше: bash deploy/start.sh"
echo "Модель: $VLM_MODEL"
echo "Настройки: $ROOT/$ENV_FILE"
