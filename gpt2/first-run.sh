#!/bin/sh
set -eu
gpt2_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
gpt2_python=${GPT2_PYTHON:-python3}
if [ ! -x "$gpt2_root/.venv/bin/python" ]; then
    "$gpt2_python" -m venv "$gpt2_root/.venv"
fi
gpt2_requirements="$gpt2_root/.venv/.requirements-installed"
if ! cmp -s "$gpt2_root/requirements.txt" "$gpt2_requirements" ||
   ! "$gpt2_root/.venv/bin/python" -c 'import numpy, regex, safetensors' >/dev/null 2>&1; then
    "$gpt2_root/.venv/bin/python" -m pip install --disable-pip-version-check -r "$gpt2_root/requirements.txt"
    cp "$gpt2_root/requirements.txt" "$gpt2_requirements"
fi
gpt2_command=generate
case "${1:-}" in
    generate|verify|doctor|setup|pack)
        gpt2_command=$1
        shift
        ;;
esac
exec "$gpt2_root/.venv/bin/python" "$gpt2_root/gpt2.py" "$gpt2_command" "$@"
