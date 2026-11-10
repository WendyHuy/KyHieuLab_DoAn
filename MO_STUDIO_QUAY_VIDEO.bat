@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===================================================
echo     KÝ HIỆU LAB - TRỢ LÝ THU VIDEO TỰ ĐỘNG
echo ===================================================
echo.
echo Đang mở Studio thu video trên trình duyệt của bạn...
echo.
start "" "%~dp0tools\studio_quay_video.html"
echo Khi quay xong, các tệp video tải về sẽ nằm trong thư mục Downloads của bạn.
echo Bạn có thể gom các tệp video này đưa vào thư mục data\clips_moi\ để kiểm thử!
echo.
pause
