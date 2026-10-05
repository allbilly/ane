#!/bin/sh
set -eu
qwen35_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ ! -x "$qwen35_root/qwen35/.venv/bin/python" ]; then
    python3 -m venv "$qwen35_root/qwen35/.venv"
    "$qwen35_root/qwen35/.venv/bin/pip" install -r "$qwen35_root/qwen35/requirements.txt"
fi
cd "$qwen35_root"
exec "$qwen35_root/qwen35/.venv/bin/python" -m qwen35 "$@"
