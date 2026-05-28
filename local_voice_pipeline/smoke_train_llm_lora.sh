#!/usr/bin/env bash
set -euo pipefail

# Tiny LoRA/data-loader smoke. Requires CUDA/Linux.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-$ROOT_DIR/local_voice_pipeline/conf/cosyvoice3_smoke.yaml}"
export EXP_DIR="${EXP_DIR:-$ROOT_DIR/local_voice_pipeline/exp_smoke/cosyvoice3/llm_lora/torch_ddp}"
export TB_DIR="${TB_DIR:-$ROOT_DIR/local_voice_pipeline/tensorboard_smoke/cosyvoice3/llm_lora/torch_ddp}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PREFETCH="${PREFETCH:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$ROOT_DIR/local_voice_pipeline/train_llm_lora.sh"
