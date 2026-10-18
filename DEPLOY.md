# Установка на сервер (Linux + NVIDIA GPU)

Нужно: Ubuntu 22.04/24.04 (или похожий Linux), драйвер NVIDIA (`nvidia-smi` работает), Python 3.10+, ~60 ГБ диска, интернет **только на время установки**.

Модель выбирается сама по видеопамяти:

| Видеопамять | Модель |
|---|---|
| ≥ 22 ГБ (3090/4090/A5000…) | Qwen3-VL-8B-Instruct-FP8 (под неё настроены правила); альтернатива — Qwen3.5-9B-FP8 через `switch_vlm.sh` |
| 11–22 ГБ | Qwen3-VL-4B-Instruct |
| < 11 ГБ | Qwen3-VL-2B-Instruct |

Свою можно задать: `VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct bash deploy/setup.sh`.

## 1. Код на сервер

Архив `fortebank-cashier-guard.tar.gz` (без датасета) скопировать и распаковать:

```bash
scp fortebank-cashier-guard.tar.gz user@SERVER:~
ssh user@SERVER
mkdir -p ~/app && tar -xzf fortebank-cashier-guard.tar.gz -C ~/app && cd ~/app
```

## 2. Установка (один раз, 10–30 минут)

```bash
bash deploy/setup.sh
```

Ставит окружения, скачивает веса модели, детектора людей и детектора предметов OWLv2, cloudflared, создаёт `deploy/.env` с новым API-ключом. Для FP8-модели на 24 ГБ включает OWLv2 (`OBJ_DETECTOR=owlv2`, `VLM_GPU_UTIL=0.72`).

## 3. Запуск

```bash
bash deploy/start.sh
```

Поднимает модель (vLLM), сервис и публичный HTTPS-адрес. В конце печатает **публичный адрес**, **API-ключ** и ссылку на сайт. Обработка работает без интернета (`HF_HUB_OFFLINE=1`); туннель нужен только чтобы жюри достучалось.

## 4. Проверка как у жюри

```bash
bash deploy/smoke.sh                                   # синтетическое видео, локально
bash deploy/smoke.sh путь/к/видео.mov https://<адрес>   # своё видео через публичный адрес
```

Печатает статус, вердикт и время. Время должно быть заметно меньше 300 с.

## Смена модели (A/B) и откат

```bash
bash deploy/switch_vlm.sh Qwen/Qwen3-VL-8B-Instruct-FP8 ""      # другая модель (веса должны быть скачаны)
bash deploy/revert_vlm.sh                                         # вернуть предыдущую за 2–3 минуты
bash deploy/check_vlm.sh                                          # проверка: «да/нет» первым токеном + память GPU
```
Адрес не меняется. Для Qwen3.5 «размышления» выключаются автоматически (`enable_thinking: false`), иначе вероятность «да» не считается.

## 5. Обслуживание

```bash
bash deploy/status.sh            # что работает, очередь, адрес, память GPU, последние логи
bash deploy/stop.sh              # остановить всё
bash deploy/stop.sh keep-tunnel  # остановить сервис и модель, публичный адрес оставить
bash deploy/update.sh /путь/к/fortebank-cashier-guard.tar.gz   # новая версия кода: перезапуск только сервиса, модель и адрес не трогаются
```

Логи: `deploy/logs/vllm.log`, `deploy/logs/app.log`, `deploy/logs/tunnel.log`.

## 6. Детектор предметов OWLv2 на уже работающем сервере

OWLv2 (Apache-2.0) показывает, где на кадре деньги, телефон, сейф, сумка, ключи, ноутбук, дверь, чашка: даёт рамку для кадра-доказательства и дополнительные моменты для вопросов Qwen. Нарушение сам не создаёт.

```bash
bash deploy/update.sh /путь/к/fortebank-cashier-guard.tar.gz   # код с OWLv2 (OWLv2 пока выключен)
bash deploy/enable_owl.sh                                       # нужен интернет один раз
```

`enable_owl.sh` ставит `transformers`, скачивает веса (~600 МБ), проверяет их офлайн, пишет в `deploy/.env` `OBJ_DETECTOR=owlv2` и `VLM_GPU_UTIL=0.72`, перезапускает vLLM и сервис (1–3 минуты сервис не отвечает — делать до сдачи адреса жюри). **Туннель не трогает — адрес прежний.** В конце печатает размер KV-кеша vLLM, память GPU и замер OWLv2 на GPU (мс/кадр, пик памяти). Если с новыми настройками не запустилось — сам ставит `OBJ_DETECTOR=off` и прежний `VLM_GPU_UTIL` и запускает как было (повторный запуск скрипта безопасен).

Выключить: `OBJ_DETECTOR=off` и `VLM_GPU_UTIL=0.80` в `deploy/.env`, затем `bash deploy/stop.sh keep-tunnel && bash deploy/start.sh`.

## Важно для сдачи

- **Не перезапускайте туннель после того, как отдали адрес жюри** — quick tunnel при перезапуске выдаёт новый адрес. Сервис и модель можно перезапускать: `bash deploy/stop.sh keep-tunnel && bash deploy/start.sh`.
- Сервер должен работать до конца проверки (баланс аренды с запасом).
- Задачи обрабатываются по одной; 5 видео подряд — нормально, бюджет времени считается от момента загрузки.
- Очистить очередь: `bash deploy/stop.sh keep-tunnel`, удалить папку `jobs/`, `bash deploy/start.sh`.

## Если что-то не так

| Симптом | Что делать |
|---|---|
| `vLLM` не стартует, в логе `out of memory` | меньшая модель: `VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct bash deploy/setup.sh`, затем `start.sh` |
| «ВНИМАНИЕ: logprobs нет» | работать будет, но вероятности станут 0/1; обновить vLLM: `.venv-vllm/bin/pip install -U vllm` |
| Нет публичного адреса | посмотреть `deploy/logs/tunnel.log`; если у сервера есть открытый порт 8080 — можно отдать `http://IP:8080` |
| после `enable_owl.sh` нехватка памяти GPU | `OWL_VLM_GPU_UTIL=0.70 bash deploy/enable_owl.sh` или выключить OWLv2 (раздел 6) |
| Видео дольше 300 с | `deploy/logs/app.log`: строки с `t_probes`, `t_total`; уменьшить `MAX_PROBE_FRAMES=160` или `PROBE_FPS=0.5` в `deploy/.env` и перезапустить |
