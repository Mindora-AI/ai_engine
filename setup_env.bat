@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo MWIZERWA AI — Environment Setup (Windows)
echo ======================================================================

:: 1. Check Python Availability
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [1/5] Detected Python version: %PY_VER%

:: 2. Create Virtual Environment if Missing
if not exist ".venv" (
    echo [2/5] Creating virtual environment (.venv)...
    python -m venv ".venv"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo       Virtual environment created successfully.
) else (
    echo [2/5] Virtual environment (.venv) already exists.
)

:: 3. Activate Virtual Environment
echo [3/5] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

:: 4. Upgrade pip and Install Dependencies
echo [4/5] Installing core project dependencies...
python -m pip install --upgrade pip
python -m pip install -r "requirements.txt"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install requirements from requirements.txt.
    echo Note: If llama-cpp-python failed to build, install a prebuilt wheel or Visual Studio C++ Build Tools.
    exit /b 1
)

:: 5. Setup DeepKIN-AgAI from ac-ai-models
echo [5/5] Checking DeepKIN-AgAI package...
set PINNED_COMMIT=c28db6bfb6b80666583717c87e8edaa2a8f384d5

if not exist "ac-ai-models" (
    echo       Cloning ac-ai-models repository...
    git clone https://github.com/c4ir-rw/ac-ai-models.git "ac-ai-models"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to clone ac-ai-models repository.
        exit /b 1
    )
    echo       Pinning ac-ai-models to verified commit %PINNED_COMMIT%...
    git -C "ac-ai-models" checkout %PINNED_COMMIT%
) else (
    echo       Directory 'ac-ai-models' already exists. Preserving existing checkout.
)

if exist "ac-ai-models\DeepKIN-AgAI" (
    echo       Installing DeepKIN-AgAI in editable mode...
    python -m pip install -e "ac-ai-models\DeepKIN-AgAI"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install DeepKIN-AgAI package.
        exit /b 1
    )
) else (
    echo [ERROR] Could not find 'ac-ai-models\DeepKIN-AgAI' directory!
    exit /b 1
)

echo ======================================================================
echo MWIZERWA AI — Setup Complete!
echo ======================================================================
echo Next steps:
echo   1. Activate environment:      call .venv\Scripts\activate.bat
echo   2. Verify / download models:   python scripts\download_models.py
echo   3. Launch assistant:           python launch_menu.py
echo ======================================================================
