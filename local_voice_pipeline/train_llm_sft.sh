#!/usr/bin/env bash
set -euo pipefail

# Official CosyVoice3 LLM SFT command for a prepared dataset.
# Requires a CUDA-capable PyTorch environment. This starts from base llm.pt;
# it never uses llm.rl.pt and does not run GRPO/RL.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/local_voice_pipeline/data/cosyvoice3}"
CONFIG="${CONFIG:-$ROOT_DIR/local_voice_pipeline/conf/cosyvoice3_sft.yaml}"
EXP_DIR="${EXP_DIR:-$ROOT_DIR/local_voice_pipeline/exp/cosyvoice3/llm/torch_ddp}"
TB_DIR="${TB_DIR:-$ROOT_DIR/local_voice_pipeline/tensorboard/cosyvoice3/llm/torch_ddp}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH="${PREFETCH:-100}"
USE_AMP_FLAG="${USE_AMP_FLAG:---use_amp}"

export CUDA_VISIBLE_DEVICES
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
JOB_ID="${JOB_ID:-1986}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:1234}"

for required in \
  "$MODEL_DIR/llm.pt" \
  "$MODEL_DIR/flow.pt" \
  "$MODEL_DIR/hift.pt" \
  "$MODEL_DIR/CosyVoice-BlankEN" \
  "$MODEL_DIR/campplus.onnx" \
  "$MODEL_DIR/speech_tokenizer_v3.onnx" \
  "$CONFIG" \
  "$DATA_DIR/train.data.list" \
  "$DATA_DIR/dev.data.list"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 1
  fi
done

# Safety: explicit base checkpoint. Do not swap this for llm.rl.pt unless a
# reward/eval loop exists and the user intentionally starts an RL phase.
echo "Training CosyVoice3 LLM SFT from base checkpoint: $MODEL_DIR/llm.pt"

torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  --rdzv_id="$JOB_ID" --rdzv_backend="c10d" --rdzv_endpoint="$RDZV_ENDPOINT" \
  "$ROOT_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config "$CONFIG" \
  --train_data "$DATA_DIR/train.data.list" \
  --cv_data "$DATA_DIR/dev.data.list" \
  --qwen_pretrain_path "$MODEL_DIR/CosyVoice-BlankEN" \
  --onnx_path "$MODEL_DIR" \
  --model llm \
  --checkpoint "$MODEL_DIR/llm.pt" \
  --model_dir "$EXP_DIR" \
  --tensorboard_dir "$TB_DIR" \
  --ddp.dist_backend "$DIST_BACKEND" \
  --num_workers "$NUM_WORKERS" \
  --prefetch "$PREFETCH" \
  --pin_memory \
  $USE_AMP_FLAG
