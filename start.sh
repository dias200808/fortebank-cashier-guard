#!/usr/bin/env bash
# Запуск сервиса одной командой (macOS / Linux).
#
#   ./start.sh          полный режим: настоящая модель через Ollama
#   ./start.sh demo     демо-режим: посмотреть интерфейс без модели
#
# Любую настройку ниже можно переопределить из окружения, например:
#   PORT=8081 MODEL=qwen3-vl:2b ./start.sh
#   NO_BROWSER=1 ./start.sh demo      (не открывать браузер)
set -euo pipefail

# ---------------------------- Настройки ----------------------------
# Тег модели в Ollama. Точное имя проверьте на https://ollama.com/library
# (поиск qwen3-vl): нужна instruct-версия БЕЗ режима рассуждений (thinking).
# Для видеокарт с 4 ГБ памяти возьмите тег 2b.
MODEL="${MODEL:-qwen3-vl:4b}"
# Ключ доступа к API и сайту. Не используйте dev-key: полный режим с ним не стартует.
API_KEY="${API_KEY:-local-key}"
PORT="${PORT:-8080}"
VLM_WORKERS="${VLM_WORKERS:-2}"       # параллельных запросов к модели
DETECTOR="${DETECTOR:-frcnn_mobile}"  # frcnn_mobile (быстрее) или frcnn_r50v2 (точнее, нужна GPU)
DEVICE="${DEVICE:-cuda}"              # без NVIDIA GPU скрипт сам переключится на cpu
# -------------------------------------------------------------------

OLLAMA_URL="http://localhost:11434"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

say()  { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
die()  { printf '\n[ОШИБКА] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Использование:
  ./start.sh          полный режим: настоящая модель через Ollama
  ./start.sh demo     демо-режим интерфейса, модель не нужна

Настройки — в начале файла или через окружение: PORT=8081 ./start.sh demo
EOF
}

case "${1:-}" in
  "")              MODE=real ;;
  demo)            MODE=demo ;;
  -h|--help|help)  usage; exit 0 ;;
  *)               usage; die "Неизвестный режим: $1" ;;
esac

# для полного режима сразу проверяем ключ и Ollama, до долгой установки
if [ "$MODE" = real ]; then
  [ "$API_KEY" != "dev-key" ] || die "API_KEY=dev-key запрещён в полном режиме. Задайте другой ключ."
  command -v ollama >/dev/null 2>&1 || die "Ollama не найдена. Установите её: https://ollama.com/download — и запустите скрипт снова. Посмотреть интерфейс без модели: ./start.sh demo"
fi

# ---------------------- 1. Окружение Python ------------------------
find_python() {
  local c
  for c in python3.12 python3 python; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      printf '%s' "$c"
      return 0
    fi
  done
  return 1
}

if [ ! -x "$PY" ]; then
  step "Первый запуск: создаю окружение Python в .venv"
  PYBOOT="$(find_python)" || die "Не найден Python 3.12. Установите его (https://www.python.org/downloads/ или brew install python@3.12) и запустите скрипт снова."
  if ! "$PYBOOT" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    say "[!] Найден $("$PYBOOT" --version 2>&1), рекомендуется Python 3.12. Продолжаю."
  fi
  "$PYBOOT" -m venv .venv || die "Не удалось создать .venv. На Debian/Ubuntu установите пакет python3.12-venv."
fi

have_modules() {
  "$PY" -c 'import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(n) for n in sys.argv[1:]) else 1)' "$@"
}

if ! have_modules torch torchvision fastapi uvicorn python_multipart av cv2 numpy PIL yaml httpx supervision trackers; then
  step "Устанавливаю зависимости (один раз, нужен интернет, это может занять 5-20 минут)"
  if ! have_modules torch torchvision; then
    if [ "$(uname -s)" = Darwin ]; then
      "$PY" -m pip install torch torchvision || die "Не удалось установить PyTorch. Проверьте интернет и запустите скрипт снова."
    elif command -v nvidia-smi >/dev/null 2>&1; then
      "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
        || die "Не удалось установить PyTorch с CUDA. Проверьте интернет и запустите скрипт снова."
    else
      "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        || die "Не удалось установить PyTorch. Проверьте интернет и запустите скрипт снова."
    fi
  fi
  "$PY" -m pip install -r requirements.txt || die "Не удалось установить зависимости из requirements.txt. Проверьте интернет и запустите скрипт снова."
fi

# ------------------------- 2. Режим работы -------------------------
http_ok() {
  if command -v curl >/dev/null 2>&1; then
    curl -sf -m 2 -o /dev/null "$1"
  else
    "$PY" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' "$1" >/dev/null 2>&1
  fi
}

case "$MODE" in
  demo)
    [ -f app/demo.py ] || die "В этой версии нет демо-режима (app/demo.py). Обновите проект."
    export PIPELINE=demo
    step "Демо-режим: модель не используется, результаты — пример, а не анализ"
    ;;
  real)
    export PIPELINE=real
    step "Проверяю Ollama и модель"
    if ! http_ok "$OLLAMA_URL/api/tags"; then
      OLLAMA_LOG="${TMPDIR:-/tmp}"
      OLLAMA_LOG="${OLLAMA_LOG%/}/ollama-serve.log"
      say "Сервер Ollama не запущен, запускаю в фоне (лог: $OLLAMA_LOG)"
      nohup ollama serve >"$OLLAMA_LOG" 2>&1 &
      for _ in $(seq 1 20); do
        http_ok "$OLLAMA_URL/api/tags" && break
        sleep 1
      done
      http_ok "$OLLAMA_URL/api/tags" || die "Сервер Ollama не ответил за 20 секунд. Запустите приложение Ollama вручную и повторите."
    fi
    say "Ollama работает."

    if ollama list 2>/dev/null | awk 'NR > 1 {print $1}' | grep -Fxq -e "$MODEL" -e "$MODEL:latest"; then
      say "Модель $MODEL уже скачана."
    else
      step "Скачиваю модель $MODEL (один раз, несколько ГБ)"
      ollama pull "$MODEL" || die "Не удалось скачать модель $MODEL. Проверьте тег на https://ollama.com/library и интернет."
    fi

    if [ "$DEVICE" = cuda ]; then
      step "Проверяю видеокарту"
      rc=0
      "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)' || rc=$?
      if [ "$rc" -eq 3 ]; then
        say "[!] Видеокарта NVIDIA с CUDA не найдена — детектор людей будет работать на процессоре (медленнее)."
        DEVICE=cpu
      elif [ "$rc" -ne 0 ]; then
        die "PyTorch не загружается (ошибка выше). Удалите папку .venv и запустите скрипт снова."
      else
        say "CUDA доступна."
      fi
    fi

    export VLM_URL="$OLLAMA_URL/v1" VLM_MODEL="$MODEL" VLM_WORKERS DEVICE DETECTOR
    ;;
esac
export API_KEY

# --------------------------- 3. Запуск -----------------------------
if ! "$PY" -c 'import socket, sys; s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1])))' "$PORT" 2>/dev/null; then
  die "Порт $PORT занят. Возможно, сервис уже запущен в другом окне: http://127.0.0.1:$PORT/?key=$API_KEY — или запустите на другом порту: PORT=8081 ./start.sh ${1:-}"
fi

URL="http://127.0.0.1:$PORT/?key=$API_KEY"

open_browser() {
  # ждём, пока сервис ответит (в полном режиме загрузка моделей занимает до минуты)
  for _ in $(seq 1 300); do
    if http_ok "http://127.0.0.1:$PORT/health"; then
      if [ "$(uname -s)" = Darwin ]; then
        open "$URL"
      elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL"
      fi
      return 0
    fi
    sleep 1
  done
}

OPENER_PID=""
if [ -z "${NO_BROWSER:-}" ]; then
  open_browser >/dev/null 2>&1 &
  OPENER_PID=$!
fi
trap '[ -n "$OPENER_PID" ] && kill "$OPENER_PID" 2>/dev/null; true' EXIT

step "Запускаю сервис"
say "Адрес:      $URL"
say "Ключ API:   $API_KEY (заголовок X-API-Key)"
say "Остановить: Ctrl+C"
[ -n "$OPENER_PID" ] && say "Браузер откроется сам, когда сервис будет готов."
say ""

rc=0
"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" || rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 130 ]; then
  die "Сервис завершился с ошибкой (код $rc). Сообщение об ошибке — выше."
fi
say "Сервис остановлен."
