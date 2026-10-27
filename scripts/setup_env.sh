#!/usr/bin/env bash
# EVOCURE - Local Virtual Environment Setup Script (Linux / macOS)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "  EVOCURE - Isolated Local Environment Setup (.venv)"
echo "=========================================================="

if command -v uv >/dev/null 2>&1; then
    echo "[1/4] Detected 'uv' package manager."
    if [ ! -x ".venv/bin/python" ]; then
        echo "[2/4] Creating isolated .venv with Python 3.12 via uv..."
        uv venv --python 3.12 --seed .venv || uv venv --python 3.11 --seed .venv
    fi
    .venv/bin/python --version
    echo "[4/4] Installing local scientific dependencies strictly into .venv..."
    uv pip install --python .venv/bin/python -r requirements.txt
else
    PYTHON_BIN=""
    for candidate in python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c "import encodings, venv" >/dev/null 2>&1; then
                PYTHON_BIN="$(command -v "$candidate")"
                break
            fi
        fi
    done

    if [ -z "$PYTHON_BIN" ]; then
        echo "Error: No working Python 3.11 or 3.12 interpreter found." >&2
        exit 1
    fi

    echo "[1/4] Validated Python: $PYTHON_BIN ($($PYTHON_BIN --version))"
    if [ ! -x ".venv/bin/python" ]; then
        echo "[2/4] Creating isolated .venv..."
        "$PYTHON_BIN" -m venv .venv
    fi
    echo "[3/4] Upgrading pip/wheel inside .venv..."
    .venv/bin/python -m pip install --upgrade pip wheel setuptools --disable-pip-version-check
    echo "[4/4] Installing local scientific dependencies..."
    .venv/bin/python -m pip install -r requirements.txt --disable-pip-version-check
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
fi

echo ""
echo "Setup complete! Run:"
echo "  ./.venv/bin/python run.py"
