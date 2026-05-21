@echo off
cd /d "%~dp0"
echo Starting Cardiovascular Risk Predictor...
echo.
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
echo.
echo Streamlit stopped. Press any key to close this window.
pause >nul
