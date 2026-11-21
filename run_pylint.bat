@echo off
REM Runs pylint over the project (read-only: it never modifies source files).
REM Output: pylint_report.txt
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run deploy_pyvenv.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" -m pylint --rcfile=.pylintrc core gui utils data main.py run_ocr.py verify_models.py > pylint_report.txt
type pylint_report.txt
echo.
echo Report written to pylint_report.txt
