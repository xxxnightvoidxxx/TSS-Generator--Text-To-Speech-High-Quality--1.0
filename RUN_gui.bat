@echo off
setlocal
pushd "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    popd
    exit /b 1
)

call .venv\Scripts\activate.bat

if not exist "gui.py" (
    echo [ERROR] gui.py not found in: %CD%
    pause
    popd
    exit /b 1
)

python gui.py
popd