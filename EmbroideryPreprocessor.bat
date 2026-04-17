@echo off
cd /d "%~dp0"

:: Load API key from .env file if it exists (skip comments and empty lines)
if exist .env (
    for /f "usebackq eol=# tokens=* delims=" %%a in (".env") do (
        if not "%%a"=="" set "%%a"
    )
)

:: Try venv python first, fall back to system python
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe launcher.py
) else (
    python launcher.py
)

if errorlevel 1 (
    echo.
    echo Something went wrong. Make sure you have set up the virtual environment:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    echo.
    pause
)
