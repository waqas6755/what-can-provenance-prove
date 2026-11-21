@echo off
cd /d "%~dp0"
echo === G5: pre-flight check ===
".venv\Scripts\python.exe" g5_preflight.py > preflight.txt 2>&1
set RC=%errorlevel%
type preflight.txt
if not "%RC%"=="0" (
  echo Pre-flight failed. Fix the FAIL lines above, then run this file again.
  pause
  exit /b 1
)
echo.
echo === G5: experiment - about 10-15 minutes, keep the computer plugged in ===
".venv\Scripts\python.exe" p3gpu_g5.py
if errorlevel 1 (
  echo G5 stopped with an error. The log is in results_g5\log.txt
  pause
  exit /b 1
)
echo.
echo === G5: validation and analysis ===
".venv\Scripts\python.exe" g5_analyse.py results_g5 results_gpu\g3_results.json
echo.
powershell -NoProfile -Command "Compress-Archive -Force -Path results_g5\* -DestinationPath results_g5.zip"
echo Finished. Claude can read the results from this folder.
pause
