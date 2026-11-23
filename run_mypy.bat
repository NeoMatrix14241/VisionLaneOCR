@echo off
REM Static type check (settings in mypy.ini). Read-only.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run deploy_pyvenv.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" -m mypy
