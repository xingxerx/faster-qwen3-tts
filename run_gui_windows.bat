@echo off
setlocal
REM Launch the Qwen3 TTS Studio GUI (Windows)
REM Usage: .\run_gui_windows.bat [model-id]
REM   e.g. .\run_gui_windows.bat Qwen/Qwen3-TTS-12Hz-0.6B-Base

set "DIR=%~dp0"
cd /d "%DIR%"

set "PY=.\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo ERROR: venv not found. Run .\setup_windows.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    "%PY%" gui\app.py
) else (
    "%PY%" gui\app.py --model %1
)
if errorlevel 1 pause
