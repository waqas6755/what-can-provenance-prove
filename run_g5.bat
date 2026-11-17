@echo off
cd /d "%~dp0"
echo Running G5 (revision experiment, about 10-15 minutes)...
".venv\Scripts\python.exe" p3gpu_g5.py
echo.
echo Finished. You can close this window.
pause
