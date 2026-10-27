#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -x ".venv/bin/python" ]; then
    bash scripts/setup_env.sh
fi
exec .venv/bin/python run.py "$@"
