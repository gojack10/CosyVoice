#!/usr/bin/env bash
set -euo pipefail

# Create a Python 3.10 macOS/Apple Silicon env for data prep and base inference.
# uv needs --no-build-isolation for old openai-whisper/pyworld packages whose
# build metadata omits pkg_resources/numpy.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
uv venv --python 3.10 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install 'setuptools<81' wheel packaging numpy==1.26.4 cython
uv pip install --no-build-isolation -r local_voice_pipeline/requirements-macos-local.txt
python local_voice_pipeline/preflight.py
