@echo off
REM QuantFunc WebUI Windows launcher
REM Double-click this file or run from cmd.

setlocal
cd /d "%~dp0"

echo === QuantFunc WebUI ===
echo.
echo Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in PATH. Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python -c "import gradio, fastapi, requests, pydantic" 2>nul
if errorlevel 1 (
    echo Installing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
)

echo.
echo Starting QuantFunc WebUI on http://127.0.0.1:7860
echo Press Ctrl+C to stop.
echo.
python app.py

pause
