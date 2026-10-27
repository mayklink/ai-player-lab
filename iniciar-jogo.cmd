@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" -m ai_player_lab browser-play --enable-input %*
if errorlevel 1 pause
