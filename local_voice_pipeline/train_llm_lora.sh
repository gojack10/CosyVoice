#!/usr/bin/env bash
set -euo pipefail

# CosyVoice3 LLM LoRA/adapter training command.
# Requires CUDA/Linux plus peft. Starts from base llm.pt; refuses llm.rl.pt.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/local_voice_pipeline/data/cosyvoice3}"
CONFIG="${CONFIG:-$ROOT_DIR/local_voice_pipeline/conf/cosyvoice3_sft.yaml}"
EXP_DIR="${EXP_DIR:-$ROOT_DIR/local_voice_pipeline/exp/cosyvoice3/llm_lora/torch_ddp}"
TB_DIR="${TB_DIR:-$ROOT_DIR/local_voice_pipeline/tensorboard/cosyvoice3/llm_lora/torch_ddp}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH="${PREFETCH:-100}"
USE_AMP_FLAG="${USE_AMP_FLAG:---use_amp}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,down_proj}"

export CUDA_VISIBLE_DEVICES
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
JOB_ID="${JOB_ID:-1986}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:1235}"

for required in \
  "$MODEL_DIR/llm.pt" \
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

echo "Training CosyVoice3 LLM LoRA from base checkpoint: $MODEL_DIR/llm.pt"

torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  --rdzv_id="$JOB_ID" --rdzv_backend="c10d" --rdzv_endpoint="$RDZV_ENDPOINT" \
  "$ROOT_DIR/local_voice_pipeline/train_lora.py" \
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
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --adapter_metadata "local_voice_pipeline first adapter run" \
  $USE_AMP_FLAG
