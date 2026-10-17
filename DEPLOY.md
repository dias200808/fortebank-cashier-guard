# Установка на сервер (Linux + NVIDIA GPU)

Нужно: Ubuntu 22.04/24.04 (или похожий Linux), драйвер NVIDIA (`nvidia-smi` работает), Python 3.10+, ~60 ГБ диска, интернет **только на время установки**.

Модель выбирается сама по видеопамяти:

| Видеопамять | Модель |
|---|---|
| ≥ 22 ГБ (3090/4090/A5000…) | Qwen3-VL-8B-Instruct (FP8, если недоступно — обычная) |
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

Ставит окружения, скачивает веса модели и детектора, cloudflared, создаёт `deploy/.env` с новым API-ключом.

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

## 5. Обслуживание

```bash
bash deploy/status.sh            # что работает, очередь, адрес, память GPU, последние логи
bash deploy/stop.sh              # остановить всё
bash deploy/stop.sh keep-tunnel  # остановить сервис и модель, публичный адрес оставить
```

Логи: `deploy/logs/vllm.log`, `deploy/logs/app.log`, `deploy/logs/tunnel.log`.

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
| Видео дольше 300 с | `deploy/logs/app.log`: строки с `t_probes`, `t_total`; уменьшить `MAX_PROBE_FRAMES=160` или `PROBE_FPS=0.5` в `deploy/.env` и перезапустить |
