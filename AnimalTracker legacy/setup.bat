@echo off
echo ========================================
echo AnimalTracker Setup
echo ========================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

echo [3/3] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application:
echo   1. Activate: venv\Scripts\activate.bat
echo   2. Run: streamlit run app.py
echo.
pause
