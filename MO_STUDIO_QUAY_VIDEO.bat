@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo     🎥 KÝ HIỆU LAB - TRỢ LÝ THU VIDEO TỰ ĐỘNG VÀO DATASET
echo ========================================================
echo.
echo Đang khởi động máy chủ quay video trực tiếp...
echo (Video bạn quay sẽ tự động lưu thẳng vào thư mục data\clips_moi\)
echo.

where py >nul 2>&1
if not errorlevel 1 (
    py tools\studio_server.py
    goto end
)

where python >nul 2>&1
if not errorlevel 1 (
    python tools\studio_server.py
    goto end
)

echo [Chế độ cục bộ]: Không tìm thấy lệnh Python, đang mở trang quay trên trình duyệt...
start "" "%~dp0tools\studio_quay_video.html"

:end
pause
