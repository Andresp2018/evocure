@echo off
title EVOCURE - Evolutionary Chemical Universe ^& Cancer Research Engine (Godot 4)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [EVOCURE] Local .venv not yet configured. Running setup...
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
)
echo [EVOCURE] Starting local scientific server and Godot 4 Bioluminescent Frontend...
".venv\Scripts\python.exe" run.py %*
pause
