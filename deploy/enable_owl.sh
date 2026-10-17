#!/usr/bin/env bash
# Turn on the OWLv2 object detector on an ALREADY RUNNING server (setup.sh + start.sh were run earlier).
# Needs internet once (pip + ~600 MB of weights). Restarts vLLM with a smaller GPU share and the service; the
# cloudflared tunnel keeps running, so the public URL stays the same. If the new start fails, it rolls back.
# Usage: bash deploy/enable_owl.sh      (OWL_VLM_GPU_UTIL=0.70 bash deploy/enable_owl.sh to pick another share)
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE="deploy/.env"
[ -f "$ENV_FILE" ] || { echo "Нет $ENV_FILE — сначала bash deploy/setup.sh"; exit 1; }
[ -f app/pipeline/objects.py ] || { echo "В коде нет OWLv2 — сначала bash deploy/update.sh /путь/к/архиву.tar.gz"; exit 1; }
set -a; . "$ENV_FILE"; set +a
OWL_MODEL="${OBJ_MODEL:-google/owlv2-base-patch16-ensemble}"
NEW_UTIL="${OWL_VLM_GPU_UTIL:-0.72}"
# Rollback target: the share before OWLv2. On a re-run (e.g. OWL_VLM_GPU_UTIL=0.70 after an OOM) or after setup.sh
# already enabled it, the current share is an OWLv2 one, so fall back to the pre-OWLv2 default instead.
if [ "${OBJ_DETECTOR:-off}" = owlv2 ]; then OLD_UTIL=0.80; else OLD_UTIL="${VLM_GPU_UTIL:-0.80}"; fi

say() { printf '\n=== %s\n' "$*"; }
setenv() {  # KEY VALUE: replace the line in deploy/.env or append it
  if grep -q "^$1=" "$ENV_FILE"; then sed -i.bak "s|^$1=.*|$1=$2|" "$ENV_FILE" && rm -f "$ENV_FILE.bak"; else echo "$1=$2" >> "$ENV_FILE"; fi
}
gpu_mem() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1; }
restart() {  # stop vLLM + service (tunnel stays), wait until vLLM has freed the GPU, start again
  bash deploy/stop.sh keep-tunnel
  for _ in $(seq 1 90); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' || true)
    [ "${USED:-0}" -lt 1500 ] && break
    sleep 1
  done
  bash deploy/start.sh
}

case "$VLM_MODEL" in
  *8B-Instruct) echo "$VLM_MODEL (bf16) занимает почти всю видеопамять — OWLv2 не поместится, нужна FP8-модель"; exit 1;;
esac

say "Зависимости сервиса (transformers)"
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import sys, torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__); \
sys.exit(0 if transformers.is_torch_available() else 'torch слишком старый для transformers (нужен >= 2.5) — OWLv2 не включён')"

say "Веса OWLv2: $OWL_MODEL"
.venv/bin/python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.txt', 'model.safetensors']))" "$OWL_MODEL"

say "Проверка офлайн (CPU, пустой кадр; GPU пока занят)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python - "$OWL_MODEL" <<'PY'
import sys
import numpy as np
from app.pipeline.objects import OwlDetector
print("OK, детекций на пустом кадре:", len(OwlDetector(sys.argv[1], "cpu").detect([np.zeros((848, 464, 3), np.uint8)])[0]))
PY

say "deploy/.env: OBJ_DETECTOR=owlv2, VLM_GPU_UTIL=$NEW_UTIL (было ${VLM_GPU_UTIL:-—})"
[ "${OBJ_DETECTOR:-off}" = owlv2 ] || cp "$ENV_FILE" "$ENV_FILE.before-owl"     # a re-run keeps the pre-OWLv2 copy
setenv OBJ_DETECTOR owlv2
setenv VLM_GPU_UTIL "$NEW_UTIL"

say "Перезапуск vLLM и сервиса (туннель остаётся, адрес прежний). Память до: $(gpu_mem)"
if ! restart; then
  say "С OWLv2 не запустилось — OBJ_DETECTOR=off, VLM_GPU_UTIL=$OLD_UTIL, запускаю как было"
  setenv OBJ_DETECTOR off
  setenv VLM_GPU_UTIL "$OLD_UTIL"
  restart
  exit 1
fi

say "Проверка"
if grep -q "object detector .* disabled" deploy/logs/app.log; then
  echo "ВНИМАНИЕ: OWLv2 не загрузился, сервис работает без него (подробности в deploy/logs/app.log)"
else
  echo "OWLv2 загружен"
fi
grep -E "KV cache size|Maximum concurrency" deploy/logs/vllm.log | tail -2 || true

say "OWLv2 на GPU (fp16, пачка 4, как в сервисе): скорость и пик памяти, отдельный процесс на ~10 с"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python - "$OWL_MODEL" "${DEVICE:-cuda}" <<'PY' \
  || echo "ВНИМАНИЕ: OWLv2 на GPU не отработал (вывод выше); сервис при такой ошибке просто пропускает этап"
import sys, time
import numpy as np
import torch
from app.pipeline.objects import OwlDetector
det = OwlDetector(sys.argv[1], sys.argv[2])
imgs = [np.random.default_rng(0).integers(0, 255, (848, 464, 3), dtype=np.uint8)] * 4
det.detect(imgs)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.monotonic()
for _ in range(5):
    det.detect(imgs)
torch.cuda.synchronize(); dt = time.monotonic() - t
print(f"{dt / 20 * 1000:.0f} мс/кадр (200 кадров ~ {dt * 10:.0f} с), "
      f"пик {torch.cuda.max_memory_allocated() / 2**30:.2f} ГиБ (веса + активации, без CUDA-контекста)")
PY
echo "Память GPU (занято, всего): $(gpu_mem)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
echo "Адрес прежний: $(cat deploy/run/public_url 2>/dev/null || echo '—')"
echo "Выключить: OBJ_DETECTOR=off и VLM_GPU_UTIL=$OLD_UTIL в $ENV_FILE, затем bash deploy/stop.sh keep-tunnel && bash deploy/start.sh"
