#!/usr/bin/env bash
# End-to-end check without the dataset: makes a 6-second synthetic .mov, sends it through the
# API exactly like the jury (X-API-Key), waits for the result and prints status + time.
# Usage: bash deploy/smoke.sh [path/to/video.mov] [base_url]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . deploy/.env; set +a
BASE="${2:-http://127.0.0.1:$PORT}"
VIDEO="${1:-}"
mkdir -p deploy/run
if [ -z "$VIDEO" ]; then
  VIDEO=deploy/run/smoke.mov
  .venv/bin/python - "$VIDEO" <<'PY'
import sys, av, numpy as np
c = av.open(sys.argv[1], "w"); s = c.add_stream("mpeg4", rate=30); s.width, s.height, s.pix_fmt = 464, 848, "yuv420p"
for i in range(180):
    img = np.full((848, 464, 3), 150, np.uint8); img[300:500, 150:300] = (90, 60, 40)
    for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")): c.mux(p)
for p in s.encode(): c.mux(p)
c.close()
PY
fi
T0=$(date +%s)
JOB=$(curl -s -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $API_KEY" -F "file=@$VIDEO" | .venv/bin/python -c "import sys, json; print(json.load(sys.stdin)['job_id'])")
echo "job_id: $JOB"
while true; do
  R=$(curl -s "$BASE/api/v1/jobs/$JOB" -H "X-API-Key: $API_KEY")
  S=$(echo "$R" | .venv/bin/python -c "import sys, json; print(json.load(sys.stdin)['status'])")
  [ "$S" != processing ] && break
  [ $(( $(date +%s) - T0 )) -gt 330 ] && { echo "СЛИШКОМ ДОЛГО (>330 с) — жюри засчитает как необработанное"; exit 1; }
  sleep 3
done
echo "$R" | .venv/bin/python -c "
import sys, json; d = json.load(sys.stdin)
print('статус:', d['status'], '| вердикт:', d.get('verdict'), '| нарушений:', len(d.get('violations', [])), '| ошибка:', d.get('error'))
for v in d.get('violations', []): print(' -', v['timecode_sec'], 'с |', v['type'], '| кадров:', len(v['evidence_frames']))"
echo "время: $(( $(date +%s) - T0 )) с (лимит жюри 300 с)"
