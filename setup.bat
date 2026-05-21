@echo off
REM CVD Risk Predictor - Setup Script for Windows

echo =====================================
echo ❤️ CVD Risk Predictor Setup (Windows)
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create .streamlit directory
echo 📁 Creating Streamlit configuration...
if not exist .streamlit mkdir .streamlit

REM Create secrets file if it doesn't exist
if not exist .streamlit\secrets.toml (
    echo 🔐 Creating secrets.toml template...
    copy .streamlit\secrets.example.toml .streamlit\secrets.toml
    echo.
    echo ⚠️ IMPORTANT: Edit .streamlit\secrets.toml and add your GEMINI_API_KEY
    echo    Get your key at: https://makersuite.google.com/app/apikey
    echo.
)

echo.
echo =====================================
echo ✅ Setup Complete!
echo =====================================
echo.
echo Next steps:
echo 1. Edit .streamlit\secrets.toml and add your GEMINI_API_KEY
echo 2. Run: streamlit run app.py
echo 3. Visit: http://localhost:8501
echo.
pause
