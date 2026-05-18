@echo off
REM CVD Predictor Setup Script for Windows

echo ====================================
echo CVD Risk Predictor - Setup
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/5] Installing backend dependencies...
pip install --upgrade pip
pip install -r backend\requirements.txt

echo [3/5] Installing TensorFlow and ML dependencies...
pip install tensorflow scikit-learn numpy joblib

echo [4/5] Creating necessary directories...
mkdir backend\uploads 2>nul

echo [5/5] Setting up environment...
if not exist .env (
    echo FLASK_ENV=development > .env
    echo JWT_SECRET_KEY=your-secret-key-change-in-production >> .env
    echo GEMINI_API_KEY= >> .env
)

echo.
echo ====================================
echo Setup Complete!
echo ====================================
echo.
echo To start the backend:
echo   1. Run: venv\Scripts\activate.bat
echo   2. Run: python -m backend.app
echo.
echo To start the frontend:
echo   Open frontend\index.html in your browser
echo   OR run a local server in the frontend directory
echo.
echo API will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:8000 (if using a server)
echo.

pause
