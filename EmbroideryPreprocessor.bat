@echo off
cd /d "%~dp0"
call venv\Scripts\activate

:: Load API key from .env file if it exists (skip comments)
if exist .env (
    for /f "usebackq eol=# tokens=* delims=" %%a in (".env") do set "%%a"
)

python launcher.py
