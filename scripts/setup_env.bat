@echo off
REM EVOCURE - Local Virtual Environment Setup (Windows CMD)
powershell -ExecutionPolicy Bypass -File "%~dp0setup_env.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo Error during .venv configuration.
    pause
    exit /b %ERRORLEVEL%
)
