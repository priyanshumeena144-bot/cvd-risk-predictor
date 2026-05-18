@echo off
REM =====================================
REM CVD PREDICTOR 3D - Quick Setup Script
REM =====================================

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║     CVD PREDICTOR 3D - SETUP SCRIPT                       ║
echo ║     Modern 3D Website with No Glitches                    ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install dependencies
echo Installing dependencies...
cd /d "c:\Users\ASUS\OneDrive\Desktop\CCVD_Project"
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully
echo.

REM Start backend
echo Starting CVD Predictor Backend...
echo.
echo 🚀 Backend server starting on http://localhost:5000
echo.
echo ═══════════════════════════════════════════════════════════
echo   NEXT: Open your browser and go to:
echo   📂 c:\Users\ASUS\OneDrive\Desktop\CCVD_Project\frontend\index.html
echo ═══════════════════════════════════════════════════════════
echo.

python backend/app.py

pause
