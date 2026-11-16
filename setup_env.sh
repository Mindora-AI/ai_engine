#!/usr/bin/env bash
# ==============================================================================
# MWIZERWA AI — Environment Setup (Linux / macOS)
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "MWIZERWA AI — Environment Setup (Linux / macOS)"
echo "======================================================================"

# 1. Detect Python
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "[ERROR] Python 3 was not found in PATH."
    echo "Please install Python 3.10, 3.11, or 3.12."
    exit 1
fi

PY_VER=$($PYTHON_BIN --version 2>&1 | cut -d' ' -f2)
echo "[1/5] Detected Python: $PYTHON_BIN ($PY_VER)"

# 2. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[2/5] Creating virtual environment (.venv)..."
    $PYTHON_BIN -m venv .venv
    echo "      Virtual environment created successfully."
else
    echo "[2/5] Virtual environment (.venv) already exists."
fi

# 3. Activate Virtual Environment
echo "[3/5] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# 4. Upgrade pip and Install Dependencies
echo "[4/5] Installing core project dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 5. Setup DeepKIN-AgAI from ac-ai-models
echo "[5/5] Checking DeepKIN-AgAI package..."
PINNED_COMMIT="c28db6bfb6b80666583717c87e8edaa2a8f384d5"

if [ ! -d "ac-ai-models" ]; then
    echo "      Cloning ac-ai-models repository..."
    git clone https://github.com/c4ir-rw/ac-ai-models.git ac-ai-models
    echo "      Pinning ac-ai-models to verified commit $PINNED_COMMIT..."
    git -C ac-ai-models checkout "$PINNED_COMMIT"
else
    echo "      Directory 'ac-ai-models' already exists. Preserving existing checkout."
fi

if [ -d "ac-ai-models/DeepKIN-AgAI" ]; then
    echo "      Installing DeepKIN-AgAI in editable mode..."
    python -m pip install -e ac-ai-models/DeepKIN-AgAI
else
    echo "[ERROR] Could not find 'ac-ai-models/DeepKIN-AgAI' directory!"
    exit 1
fi

echo "======================================================================"
echo "MWIZERWA AI — Setup Complete!"
echo "======================================================================"
echo "Next steps:"
echo "  1. Activate environment:      source .venv/bin/activate"
echo "  2. Verify / download models:   python scripts/download_models.py"
echo "  3. Launch assistant:           python launch_menu.py"
echo "======================================================================"
