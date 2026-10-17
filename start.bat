@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title Контроль кассовой дисциплины
rem Запуск сервиса одной командой (Windows).
rem
rem   start.bat          полный режим: настоящая модель через Ollama
rem   start.bat demo     демо-режим: посмотреть интерфейс без модели
rem
rem Двойной клик по файлу запускает полный режим.

rem ============================ Настройки ============================
rem Тег модели в Ollama. Точное имя проверьте на https://ollama.com/library
rem в поиске qwen3-vl: нужна instruct-версия БЕЗ режима рассуждений, не thinking.
rem Для видеокарт с 4 ГБ памяти возьмите тег 2b, например qwen3-vl:2b
set "MODEL=qwen3-vl:4b"
rem Ключ доступа к API и сайту: только латиница, цифры и дефис.
rem Не используйте dev-key: полный режим с ним не стартует.
set "API_KEY=local-key"
set "PORT=8080"
rem Сколько запросов к модели отправлять параллельно; для ноутбука 2.
set "VLM_WORKERS=2"
rem Детектор людей: frcnn_mobile быстрее, frcnn_r50v2 точнее, но нужна мощная видеокарта.
set "DETECTOR=frcnn_mobile"
rem cuda - видеокарта NVIDIA. Если её нет, скрипт сам переключится на cpu.
set "DEVICE=cuda"
rem Формат координат модели, см. docs\windows_laptop.md, раздел 3. Обычно не нужен.
rem set "VLM_COORD_MODE=rel1000"
rem ===================================================================

pushd "%~dp0"
if errorlevel 1 (
  echo [ОШИБКА] Не удалось перейти в папку проекта.
  pause
  exit /b 1
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=real"
if /i "%MODE%"=="real" goto mode_known
if /i "%MODE%"=="demo" goto mode_known
if /i "%MODE%"=="help" goto usage
if /i "%MODE%"=="-h" goto usage
if /i "%MODE%"=="--help" goto usage
echo [ОШИБКА] Неизвестный режим: %MODE%
echo.
call :print_usage
goto end_fail

:usage
call :print_usage
goto end_ok

:mode_known
echo.
echo Контроль кассовой дисциплины - локальный сервис
if /i "%MODE%"=="real" echo Режим: полный, модель %MODEL% через Ollama
if /i "%MODE%"=="demo" echo Режим: демо, без модели

rem Для полного режима сразу проверяем ключ и Ollama, до долгой установки.
if /i not "%MODE%"=="real" goto check_python
if /i "%API_KEY%"=="dev-key" goto fail_devkey
where ollama >nul 2>&1
if not errorlevel 1 goto check_python
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
where ollama >nul 2>&1
if errorlevel 1 goto fail_noollama

rem ---------------------- 1. Окружение Python ----------------------
:check_python
set "PY=.venv\Scripts\python.exe"
if exist "%PY%" goto venv_ok

echo.
echo [1/3] Первый запуск: создаю окружение Python в папке .venv
set "PYBOOT="
call :check_py "py -3.12"
if not errorlevel 1 set "PYBOOT=py -3.12"
if defined PYBOOT goto make_venv
call :check_py "python"
if not errorlevel 1 set "PYBOOT=python"
if defined PYBOOT goto make_venv
call :check_py "py -3"
if not errorlevel 1 set "PYBOOT=py -3"
if defined PYBOOT goto make_venv
goto fail_nopython

:make_venv
%PYBOOT% -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if errorlevel 1 echo [!] Найден Python не версии 3.12. Рекомендуется 3.12, продолжаю.
%PYBOOT% -m venv .venv
if errorlevel 1 goto fail_venv
if not exist "%PY%" goto fail_venv

:venv_ok
"%PY%" -c "import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(n) for n in sys.argv[1:]) else 1)" torch torchvision fastapi uvicorn python_multipart av cv2 numpy PIL yaml httpx supervision trackers >nul 2>&1
if not errorlevel 1 goto deps_ok

echo.
echo [2/3] Устанавливаю зависимости. Это делается один раз, нужен интернет, займёт 5-20 минут.
"%PY%" -c "import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(n) for n in sys.argv[1:]) else 1)" torch torchvision >nul 2>&1
if not errorlevel 1 goto torch_ok
echo Скачиваю PyTorch с поддержкой CUDA, около 3 ГБ...
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail_torch
:torch_ok
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail_deps
:deps_ok

rem ------------------------- 2. Режим работы ------------------------
if /i "%MODE%"=="demo" goto mode_demo
goto mode_real

:mode_demo
if not exist "app\demo.py" goto fail_nodemo
set "PIPELINE=demo"
echo.
echo Демо-режим: модель не используется, результаты - пример, а не анализ видео.
goto launch

:mode_real
set "PIPELINE=real"
echo.
echo [3/3] Проверяю Ollama и модель
call :ollama_ping
if not errorlevel 1 goto ollama_up
echo Сервер Ollama не запущен, запускаю его в отдельном свёрнутом окне...
start "Ollama" /min ollama serve
set /a TRIES=0
:wait_ollama
ping -n 2 127.0.0.1 >nul
call :ollama_ping
if not errorlevel 1 goto ollama_up
set /a TRIES+=1
if %TRIES% lss 20 goto wait_ollama
goto fail_ollama_timeout
:ollama_up
echo Ollama работает.

set "HAVE_MODEL="
for /f "skip=1 tokens=1" %%m in ('ollama list 2^>nul') do (
  if /i "%%m"=="%MODEL%" set "HAVE_MODEL=1"
  if /i "%%m"=="%MODEL%:latest" set "HAVE_MODEL=1"
)
if defined HAVE_MODEL goto model_ok
echo Скачиваю модель %MODEL%. Это делается один раз, несколько ГБ...
ollama pull %MODEL%
if errorlevel 1 goto fail_pull
:model_ok
echo Модель %MODEL% готова.

if /i not "%DEVICE%"=="cuda" goto device_ok
echo Проверяю видеокарту...
"%PY%" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"
set "RC=%errorlevel%"
if "%RC%"=="0" goto cuda_ok
if not "%RC%"=="3" goto fail_torch_broken
echo [!] Видеокарта NVIDIA с CUDA не найдена. Детектор людей будет работать на процессоре, медленнее.
set "DEVICE=cpu"
goto device_ok
:cuda_ok
echo CUDA доступна.
:device_ok

set "VLM_URL=http://localhost:11434/v1"
set "VLM_MODEL=%MODEL%"
goto launch

rem --------------------------- 3. Запуск ----------------------------
:launch
"%PY%" -c "import socket, sys; s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1])))" %PORT% >nul 2>&1
if errorlevel 1 goto fail_port

set "URL=http://127.0.0.1:%PORT%/?key=%API_KEY%"
rem Браузер открывается в фоне этого же окна, как только сервис ответит на /health.
start "" /b powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; for ($i = 0; $i -lt 300; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri 'http://127.0.0.1:%PORT%/health' | Out-Null; Start-Process '%URL%'; break } catch { Start-Sleep -Seconds 1 } }"

echo.
echo Запускаю сервис
echo   Адрес:      %URL%
echo   Ключ API:   %API_KEY%, заголовок X-API-Key
echo   Остановить: Ctrl+C или закройте это окно
echo Браузер откроется сам, когда сервис будет готов. В полном режиме это до минуты.
echo.
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" goto stopped_ok
rem -1073741510 = 0xC000013A: процесс остановлен через Ctrl+C
if "%RC%"=="-1073741510" goto stopped_ok
echo [ОШИБКА] Сервис завершился с ошибкой, код %RC%. Причина - в сообщениях выше.
goto end_fail
:stopped_ok
echo Сервис остановлен.
goto end_ok

rem ----------------------------- Ошибки -----------------------------
:fail_nopython
echo.
echo [ОШИБКА] Не найден Python 3.12.
echo Установите его с https://www.python.org/downloads/ и отметьте пункт "Add python.exe to PATH",
echo затем запустите start.bat снова.
goto end_fail

:fail_venv
echo.
echo [ОШИБКА] Не удалось создать окружение Python в папке .venv.
echo Удалите папку .venv, если она осталась, и запустите start.bat снова.
goto end_fail

:fail_torch
echo.
echo [ОШИБКА] Не удалось установить PyTorch. Проверьте интернет и запустите start.bat снова.
goto end_fail

:fail_deps
echo.
echo [ОШИБКА] Не удалось установить зависимости из requirements.txt.
echo Проверьте интернет и запустите start.bat снова.
goto end_fail

:fail_torch_broken
echo.
echo [ОШИБКА] PyTorch не загружается, текст ошибки выше.
echo Частая причина - нет Microsoft Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo Если не помогло, удалите папку .venv и запустите start.bat снова.
goto end_fail

:fail_nodemo
echo.
echo [ОШИБКА] В этой версии нет демо-режима, файл app\demo.py не найден. Обновите проект.
goto end_fail

:fail_devkey
echo.
echo [ОШИБКА] API_KEY=dev-key запрещён в полном режиме. Задайте другой ключ в начале start.bat.
goto end_fail

:fail_noollama
echo.
echo [ОШИБКА] Ollama не найдена. Установите её: https://ollama.com/download
echo После установки закройте это окно и запустите start.bat снова.
echo Посмотреть интерфейс без модели можно так: start.bat demo
goto end_fail

:fail_ollama_timeout
echo.
echo [ОШИБКА] Сервер Ollama не ответил за 20 секунд.
echo Запустите приложение Ollama из меню Пуск и повторите.
goto end_fail

:fail_pull
echo.
echo [ОШИБКА] Не удалось скачать модель %MODEL%.
echo Проверьте интернет и точное имя тега на https://ollama.com/library, затем исправьте MODEL в начале start.bat.
goto end_fail

:fail_port
echo.
echo [ОШИБКА] Порт %PORT% занят. Возможно, сервис уже запущен в другом окне:
echo   http://127.0.0.1:%PORT%/?key=%API_KEY%
echo Закройте то окно или поменяйте PORT в начале start.bat.
goto end_fail

:end_fail
echo.
pause
popd
exit /b 1

:end_ok
echo.
pause
popd
exit /b 0

rem -------------------------- Подпрограммы --------------------------
:print_usage
echo Использование:
echo   start.bat          полный режим: настоящая модель через Ollama
echo   start.bat demo     демо-режим интерфейса, модель не нужна
echo Настройки - в начале файла start.bat, блок "Настройки".
exit /b 0

:check_py
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
exit /b %errorlevel%

:ollama_ping
where curl >nul 2>&1
if errorlevel 1 goto ollama_ping_ps
curl -s -f -m 2 -o nul http://localhost:11434/api/tags >nul 2>&1
exit /b %errorlevel%
:ollama_ping_ps
powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri 'http://localhost:11434/api/tags' | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
exit /b %errorlevel%
