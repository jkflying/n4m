#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="$ROOT/external/venv_models"

if [ ! -f "$VENV/bin/python3" ]; then
    echo "Error: $VENV not found. Set up the model export venv first." >&2
    exit 1
fi

PY="$VENV/bin/python3"

for script in \
    1_export_xfeat_onnx.py \
    2_export_reference.py \
    3_test_onnx_model.py \
    4_export_lightglue_onnx.py \
    5_export_lightglue_reference.py \
    6_test_lightglue_onnx.py
do
    echo "=== $script ==="
    "$PY" "$SCRIPT_DIR/$script"
    echo
done

echo "All export steps completed."
