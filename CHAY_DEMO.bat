@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 --version >nul 2>&1
if errorlevel 1 (
 echo Can cai Python 3.11 64-bit. Goi nay khong dung Python 3.14.
 pause
 exit /b 1
)
if not exist .venv\Scripts\python.exe py -3.11 -m venv .venv
if not exist .venv\Scripts\python.exe (
 echo Khong tao duoc moi truong. Hay giai nen toan bo ZIP vao thu muc co quyen ghi.
 pause
 exit /b 1
)
.venv\Scripts\python.exe -c "import numpy, cv2, mediapipe; assert mediapipe.__version__ == '0.10.21'" >nul 2>&1
if errorlevel 1 (
 .venv\Scripts\python.exe -m pip install -r requirements.txt
 if errorlevel 1 (
  echo Cai thu vien that bai. Lan cai dau can ket noi mang.
  pause
  exit /b 1
 )
)
if not exist .venv\khl_v6_ready (
 echo Dang chuan bi bo trich diem. Lan dau co the can mang, vui long cho...
 .venv\Scripts\python.exe tools\doctor.py --warmup > .venv\khl_prepare.log 2>&1
 if errorlevel 1 (
  type .venv\khl_prepare.log
  echo Kiem tra video chua dat. Giu cua so nay de xem loi, roi chay lai khi da xu ly.
  pause
  exit /b 1
 )
 type nul > .venv\khl_v6_ready
 echo Kiem tra dat. Dang mo ung dung...
)
.venv\Scripts\python.exe run.py
pause
