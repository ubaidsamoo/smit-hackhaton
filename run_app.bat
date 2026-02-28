@echo off
REM Flight Delay Prediction - Streamlit Deployment Script
REM Run both API and Streamlit simultaneously

echo.
echo ================================================================================
echo         FLIGHT DELAY PREDICTION - STREAMLIT DEPLOYMENT
echo ================================================================================
echo.

REM Check if API is already running
echo Checking API status...
curl -s http://127.0.0.1:8000/health > nul
if %errorlevel% equ 0 (
    echo ✓ API is already running
    goto STREAMLIT
) else (
    echo Starting API server...
    start "API Server" cmd /k python flight_delay_api.py
    echo Waiting for API to start (10 seconds)...
    timeout /t 10 /nobreak
)

:STREAMLIT
echo.
echo ✓ Starting Streamlit application...
echo.
echo ================================================================================
echo         STREAMLIT APP STARTING
echo ================================================================================
echo.
echo 🌐 Streamlit will open at: http://localhost:8501
echo 📡 API running at: http://localhost:8000
echo 📚 API Docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C in the Streamlit window to stop
echo ================================================================================
echo.

streamlit run streamlit_app.py --logger.level=error

pause
