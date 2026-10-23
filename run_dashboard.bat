@echo off
title Predictive Localization Dashboard
color 0A

echo.
echo  ================================================
echo   Predictive Localization Dashboard - Launcher
echo  ================================================
echo.

:: Navigate to the project directory
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .\venv\
    echo         Please create it with: python -m venv venv
    echo         Then install dependencies: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [ERROR] Streamlit not found in the virtual environment.
    echo         Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Please check requirements.txt.
        pause
        exit /b 1
    )
)

:: Launch the Streamlit app
echo [2/3] Starting Streamlit server...
echo [3/3] Opening dashboard in your browser...
echo.
echo  Dashboard URL: http://localhost:8501
echo  Press Ctrl+C to stop the server.
echo.

streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

:: On exit
echo.
echo  Dashboard stopped. Goodbye!
pause
