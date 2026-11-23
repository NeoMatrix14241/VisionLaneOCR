@echo off
setlocal enabledelayedexpansion

title VisionLane OCR - Virtual Environment Setup
echo ========================================
echo VisionLane OCR Virtual Environment Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.12+ and add it to your PATH
    echo.
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.12 or newer is required ^(see requirements.txt^).
    python --version
    echo.
    pause
    exit /b 1
)

echo Python found. Creating virtual environment...

REM Remove existing virtual environment if it exists
if exist ".venv" (
    echo Removing existing virtual environment...
    rmdir /s /q .venv 2>nul
)

REM Create virtual environment
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Please check your Python installation and permissions
    echo.
    pause
    exit /b 1
)

echo Virtual environment created successfully.
echo.

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Please ensure requirements.txt is in the same directory as this script
    echo.
    pause
    exit /b 1
)

echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

echo.
set "REQ_FLAGS=--upgrade"
if /i "%~1"=="gpu" (
    if not exist "requirements-torch-gpu.txt" (
        echo ERROR: requirements-torch-gpu.txt not found
        pause
        exit /b 1
    )
    echo Installing the CUDA build of PyTorch from requirements-torch-gpu.txt - this is several GB...
    .venv\Scripts\python.exe -m pip install -r requirements-torch-gpu.txt
    if errorlevel 1 (
        echo ERROR: Failed to install the CUDA build of PyTorch
        echo Check your internet connection and the CUDA tag in requirements-torch-gpu.txt
        pause
        exit /b 1
    )
    REM no --upgrade here: it could replace the CUDA build by a newer CPU wheel from PyPI
    set "REQ_FLAGS="
)
echo Installing requirements from requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt %REQ_FLAGS%
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    echo Please check your requirements.txt file and internet connection
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Virtual environment setup completed successfully!
echo ========================================
echo.
echo To activate the virtual environment manually, run:
echo   .venv\Scripts\activate
echo.
echo For an NVIDIA GPU run this script as:  deploy_pyvenv.bat gpu   ^(installs the CUDA build of PyTorch^)
echo.
echo To run the application with created virtual environment, use:
echo   .venv\Scripts\python main.py
echo.
echo To run the tests ^(needs requirements-dev.txt^):
echo   .venv\Scripts\python -m pytest
echo.
echo Or use the build script:
echo   build_nuitka.bat --> for nuitka
echo   build_pyinstaller.bat --> for pyinstaller
echo.
echo Press any key to exit...
pause >nul