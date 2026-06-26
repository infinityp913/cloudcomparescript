#!/bin/zsh
# Launcher for cloudcomparescript — sets up CloudComPy environment and runs a Python script.
# Usage: ./run.sh pre_snip_script.py
#        ./run.sh post_snip_script.py

CLOUDCOMPY_ROOT=~/Desktop/CloudComPy310_clean
CONDA_PYTHON=~/miniconda3/envs/CloudComPy310/bin/python3.10

export PYTHONPATH="${CLOUDCOMPY_ROOT}:${CLOUDCOMPY_ROOT}/CloudCompare/CloudCompare.app/Contents/Frameworks:${CLOUDCOMPY_ROOT}/doc/PythonAPI_test"

# Load .env if present (for ANTHROPIC_API_KEY etc.)
if [ -f "$(dirname "$0")/.env" ]; then
    set -a
    source "$(dirname "$0")/.env"
    set +a
fi

exec "$CONDA_PYTHON" "$@"
