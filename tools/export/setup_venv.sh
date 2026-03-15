#!/usr/bin/env bash
# Create and populate the model-export virtualenv.
# Usage: ./setup_venv.sh [--force]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$ROOT/external/venv_models"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

if [[ "${1:-}" == "--force" ]]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
    echo "Venv already exists at $VENV_DIR (use --force to recreate)"
    exit 0
fi

echo "Creating venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing dependencies from $REQUIREMENTS ..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS"

echo "Done. Activate with: source $VENV_DIR/bin/activate"
