@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   TTS Generator - Setup
echo ========================================

pushd "%~dp0"

:: ---- Check Python ----
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10 or newer.
    pause
    exit /b 1
)

:: ---- Create virtual environment ----
if not exist ".venv" (
    echo [*] Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

:: ---- Activate ----
call .venv\Scripts\activate.bat

:: ---- Upgrade pip ----
python -m pip install --upgrade pip --quiet

:: ---- Install dependencies ----
echo [*] Installing dependencies...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

:: ---- Pre-download model ----
echo [*] Pre-downloading TTS model...
python main.py --setup

:: ---- Generate run.bat ----
(
    echo @echo off
    echo pushd "%%~dp0"
    echo call .venv\Scripts\activate.bat
    echo python main.py %%*
    echo popd
) > run.bat

echo.
echo ========================================
echo   Setup complete.
echo   Usage: run.bat --text "Hello world"
echo ========================================
pause
popd