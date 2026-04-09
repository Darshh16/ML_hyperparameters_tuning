@echo off
REM Quick Start Script for ML Hyperparameter Visualizer
echo.
echo ========================================
echo   ML Hyperparameter Visualizer
echo   Streamlit Web Application
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import streamlit; import sklearn; import plotly" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install packages
        pause
        exit /b 1
    )
)

echo.
echo Starting ML Hyperparameter Visualizer...
echo.
echo The application will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

streamlit run app.py --logger.level=info

pause
