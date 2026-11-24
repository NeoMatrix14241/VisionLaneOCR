@echo off
setlocal
title App Builder: VisionLane OCR (Nuitka)
REM --------------------------------------------------------------------
REM Nuitka standalone build for VisionLane OCR (Windows).
REM
REM Prerequisites: .venv created with deploy_pyvenv.bat, then
REM     .venv\Scripts\pip install -r requirements-dev.txt
REM A C compiler is needed; Nuitka offers to download MinGW64 on first run.
REM Optional: install ccache (e.g. "winget install ccache.ccache") and put it on PATH to speed up rebuilds.
REM
REM NOTE: this script was rewritten in the refactoring session without access to Windows.
REM       Do a test build and run the .exe on a clean machine before releasing.
REM --------------------------------------------------------------------
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run deploy_pyvenv.bat first.
    pause
    exit /b 1
)

echo Making sure the default DocTR models are present...
".venv\Scripts\python.exe" verify_models.py
if errorlevel 1 (
    echo Failed to download the DocTR models. Aborting build.
    pause
    exit /b 1
)

rmdir /s /q dist_nuitka 2>nul
rem --jobs=%NUMBER_OF_PROCESSORS% ^
echo Building with Nuitka (this takes a while)...
".venv\Scripts\python.exe" -m nuitka ^
    --standalone ^
    --assume-yes-for-downloads ^
    --low-memory ^
    --output-dir=dist_nuitka ^
    --output-filename=VisionLaneOCR.exe ^
    --windows-icon-from-ico=icon.ico ^
    --windows-console-mode=disable ^
    --enable-plugin=pyqt6 ^
    --include-qt-plugins=platforms,imageformats,styles ^
    --include-package=doctr ^
    --include-package-data=doctr ^
    --include-package-data=reportlab ^
    --include-package-data=pypdfium2 ^
    --include-package-data=pypdfium2_raw ^
    --include-data-files=icon.ico=icon.ico ^
    --include-data-dir=assets=assets ^
    main.py

if errorlevel 1 (
    echo Build failed! Check the messages above.
    pause
    exit /b 1
)
echo.
echo Build complete: dist_nuitka\main.dist\VisionLaneOCR.exe
echo (Use --windows-console-mode=force temporarily if you need to see start-up errors.)
pause
