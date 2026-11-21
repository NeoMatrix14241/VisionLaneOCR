@echo off
REM Removes build output and generated reports. (Logs and settings now live in %LOCALAPPDATA%\VisionLaneOCR.)
rmdir /s /q "%~dp0dist_nuitka" 2>nul
rmdir /s /q "%~dp0dist_pyinstaller" 2>nul
rmdir /s /q "%~dp0.pytest_cache" 2>nul
del /q "%~dp0pylint_report.txt" 2>nul
echo Cleanup completed.
