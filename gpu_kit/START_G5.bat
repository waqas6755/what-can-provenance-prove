@echo off
cd /d "%~dp0"
echo Paper 3 - G5 experiment. Keep this window open until it says Finished.
echo.
nvidia-smi > gpu_info.txt 2>&1
set RC=%errorlevel%
type gpu_info.txt
if not "%RC%"=="0" (
  echo No NVIDIA GPU or driver found on this computer. G5 needs an NVIDIA GPU. Stopping.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo === First run: setting up a private Python environment - 10 to 20 minutes, needs internet ===
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_g5_env.ps1"
  if errorlevel 1 (
    echo Setup failed. Send setup_g5_transcript.txt
    pause
    exit /b 1
  )
)
call "%~dp0run_g5.bat"
