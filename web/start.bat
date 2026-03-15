@echo off
REM Quick start script for Bitcoin Trading Predictor Web Application
REM Script khởi động nhanh cho ứng dụng web

echo ========================================
echo Bitcoin Trading Predictor - Web App
echo ========================================
echo.

REM Check if conda environment exists
echo [1/3] Activating conda environment...
call conda activate MeoMeo
if errorlevel 1 (
    echo ERROR: Cannot activate MeoMeo environment
    echo Please create it first: conda create -n MeoMeo python=3.10
    pause
    exit /b 1
)

echo.
echo [2/3] Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Flask not installed. Installing dependencies...
    pip install flask flask-cors
)

echo.
echo [3/3] Starting web server...
echo.
echo ========================================
echo Server will start at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"
python app.py

pause
