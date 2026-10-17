# Запуск на Windows-ноутбуке (проверка, не для жюри)

README описывает боевой вариант: Linux-сервер с GPU 24 ГБ, vLLM и Qwen3-VL-8B. Коду 24 ГБ не нужно — это требование только модели 8B. На ноутбуке с 4–8 ГБ видеопамяти можно запустить меньшую модель через Ollama и прогнать 1–2 видео. Качество будет ниже, это проверка «всё работает с настоящей моделью», а не замер точности.

| Видеокарта | Модель | Примечание |
|---|---|---|
| 8 ГБ (RTX 5060 и т.п.) | Qwen3-VL 4B | основной вариант для ноутбука |
| 4 ГБ (RTX 3050 Ti и т.п.) | Qwen3-VL 2B | слабее, только для проверки |

Точные теги моделей смотрите на ollama.com/library (поиск `qwen3-vl`); нужна instruct-версия **без режима рассуждений** (thinking), иначе ответы «Yes/No» не распознаются. Не берите Qwen2.5-VL-3B: у неё некоммерческая лицензия.

## 1. Установка

Нужны: драйвер NVIDIA, Git, Python 3.12, [Ollama для Windows](https://ollama.com/download).

```powershell
git clone https://github.com/dias200808/fortebank-cashier-guard.git
cd fortebank-cashier-guard
py -3.12 -m venv .venv
.venv\Scripts\activate
# torch с CUDA: cu128 подходит и для RTX 50xx, и для RTX 30xx
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Скопируйте датасет организаторов в папку `open\` (`open\videos\*.mov`, `open\open_labels.xlsx`) — в репозитории его нет.

## 2. Модель

```powershell
ollama pull <тег qwen3-vl 4b или 2b>
ollama list          # модель должна появиться в списке
```

Ollama сама поднимает сервер на `http://localhost:11434`; наш код обращается к нему по OpenAI-совместимому адресу `http://localhost:11434/v1`.

## 3. Настройки (в том же окне PowerShell)

```powershell
$env:VLM_URL = "http://localhost:11434/v1"
$env:VLM_MODEL = "<тот же тег>"
$env:VLM_WORKERS = "2"          # на ноутбуке меньше параллельных запросов
$env:DEVICE = "cuda"
$env:DETECTOR = "frcnn_mobile"
$env:API_KEY = "test-key-123"
```

Проверка формата координат (один раз):

```powershell
python -m scripts.check_grounding http://localhost:11434/v1 <тег>
```

Скрипт напечатает IoU для `rel1000` и `abs`. Поставьте режим с IoU > 0.5: `$env:VLM_COORD_MODE = "rel1000"` (или `"abs"`).

## 4. Прогон 1–2 видео

```powershell
python -m scripts.eval --only X-26,X-25 --out eval_out\try1
```

X-26 — нарушение (наличные на столе при уходе, метка 9 с), X-25 — чистое видео (правильный уход). В консоли будет строка на каждое видео и метрики; подробности по кандидатам — в `eval_out\try1\<видео>.json`.

## 5. Сайт

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Откройте `http://127.0.0.1:8080/?key=test-key-123` и загрузите `.mov`.

## Что может пойти не так

- **В логе предупреждение про logprobs.** Если Ollama не возвращает вероятности токенов, ответы модели превращаются в 0/1. Работает, но грубее, чем на vLLM.
- **Медленно.** На 4–8 ГБ один ролик может обрабатываться дольше 2 минут; при нехватке времени система пропускает перепроверку (`debug.verification_skipped` в json).
- **Нехватка видеопамяти.** Уменьшите `$env:DET_BATCH = "2"`, закройте другие программы, используйте модель меньше.
- **Модель отвечает рассуждениями, а не Yes/No.** Возьмите instruct-тег без thinking.
