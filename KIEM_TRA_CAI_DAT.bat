@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 echo Hay chay CHAY_DEMO.bat truoc de cai moi truong.
 pause
 exit /b 1
)
.venv\Scripts\python.exe tools\doctor.py --warmup
pause
