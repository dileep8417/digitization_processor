@echo off
cd /d "%~dp0"

:: Load API key from .env file if it exists (skip comments and empty lines)
if exist .env (
    for /f "usebackq eol=# tokens=* delims=" %%a in (".env") do (
        if not "%%a"=="" set "%%a"
    )
)

:: Check for venv
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )
)

:: Install dependencies if missing
venv\Scripts\python.exe -c "import cv2" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    venv\Scripts\pip.exe install -r requirements.txt
)

echo Starting Embroidery Preprocessor...
venv\Scripts\python.exe launcher.py

echo.
echo Server stopped.
pause
