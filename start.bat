@echo off
REM LazySD UI Windows launcher
REM Double-click this file or run from cmd.

setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_XET=1
REM Multi-threaded Rust downloader (needs: pip install hf-transfer, see requirements.txt)
set HF_HUB_ENABLE_HF_TRANSFER=1
REM Optional mirror if huggingface.co CDN is slow. WARNING: HF_TOKEN would be sent to the mirror.
REM set HF_ENDPOINT=https://hf-mirror.com
cd /d "%~dp0"

echo === LazySD UI ===
echo.

set "PY_CMD=python"
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
)

echo Checking Python (%PY_CMD%)...
"%PY_CMD%" --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in PATH. Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
"%PY_CMD%" -c "import fastapi, uvicorn, requests, pydantic, PIL" 2>nul
if errorlevel 1 (
    echo Installing dependencies from requirements.txt...
    "%PY_CMD%" -m pip install -r requirements.txt
)

echo.
echo Starting LazySD UI on http://127.0.0.1:7860
echo Press Ctrl+C to stop.
echo.
"%PY_CMD%" app.py

pause
