#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    echo "Error: ${PYTHON_BIN} not found. Set PYTHON_BIN to an available interpreter." >&2
    exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -r "${PROJECT_ROOT}/requirements.txt"

LOG_DIR="${PROJECT_ROOT}/artifacts/logs"
mkdir -p "${LOG_DIR}"
pip freeze > "${LOG_DIR}/pip_freeze.txt"

deactivate

echo "Virtual environment created at ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
