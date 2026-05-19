@echo off
cd /d "%~dp0"
echo Pic Aliver - Starting Application...
echo Usage: %0 [--model "Model Name"]
python -m src.picture_aliver.app %*
pause
