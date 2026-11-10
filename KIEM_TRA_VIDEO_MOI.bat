@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 tools\import_new_clips.py 2>nul
if errorlevel 1 (
  python tools\import_new_clips.py
)
pause
