#!/usr/bin/env bash
# Runs farol from WSL with a Windows Python, so the model runs on the GPU through DirectML.
# WSL itself has no GPU access for most AMD cards (ROCm on WSL only supports Radeon RX 7000+).
#
# One-time setup on Windows (PowerShell):
#   py -3.11 -m venv $HOME\.farol\venv
#   $HOME\.farol\venv\Scripts\python.exe -m pip install ultralytics onnxruntime-directml opencv-python-headless lap
# and export the model in WSL:
#   uv run --with onnx --with onnxslim python -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='onnx', imgsz=960)"
#
# Usage: scripts/farol-windows.sh <video> [farol options...]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -z "${FAROL_WINDOWS_PYTHON:-}" ]]; then
    profile=$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')
    FAROL_WINDOWS_PYTHON=$(wslpath -u "$profile")/.farol/venv/Scripts/python.exe
fi
[[ -x "$FAROL_WINDOWS_PYTHON" ]] || { echo "Windows Python not found: $FAROL_WINDOWS_PYTHON (see setup at the top of $0)" >&2; exit 1; }

args=()
has_model=0
prev=
for arg in "$@"; do
    [[ "$arg" == --model ]] && has_model=1
    [[ "$prev" == --out ]] && mkdir -p "$arg"  # wslpath needs the folder to exist
    if [[ -e "$arg" ]]; then arg=$(wslpath -w "$(realpath "$arg")"); fi
    args+=("$arg")
    prev=$arg
done
(( has_model )) || args+=(--model "$(wslpath -w "$ROOT/yolo11s.onnx")")

export PYTHONPATH="$ROOT" YOLO_AUTOINSTALL=False PYTHONUTF8=1 PYTHONUNBUFFERED=1
export WSLENV="PYTHONPATH/p:YOLO_AUTOINSTALL:PYTHONUTF8:PYTHONUNBUFFERED${WSLENV:+:$WSLENV}"
cd /mnt/c  # a Windows working directory avoids cmd/UNC path issues
exec "$FAROL_WINDOWS_PYTHON" -m farol "${args[@]}"
