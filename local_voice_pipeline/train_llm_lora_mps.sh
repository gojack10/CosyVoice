#!/usr/bin/env bash
set -euo pipefail

# Apple Silicon/MPS LoRA training with granular checkpoints and observability.
# This intentionally does not use CUDA/DDP. It keeps fp32 MPS training and saves
# every optimizer step by default so we can select the best pre-overfit adapter.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/local_voice_pipeline/data/cosyvoice3}"
CONFIG="${CONFIG:-$ROOT_DIR/local_voice_pipeline/conf/cosyvoice3_sft.yaml}"
EXP_DIR="${EXP_DIR:-$ROOT_DIR/local_voice_pipeline/exp/cosyvoice3/llm_lora/mps_single}"
TB_DIR="${TB_DIR:-$ROOT_DIR/local_voice_pipeline/tensorboard/cosyvoice3/llm_lora/mps_single}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-5}"
ACCUM_GRAD="${ACCUM_GRAD:-2}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
MIN_DELTA="${MIN_DELTA:-0.0}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,down_proj}"
RESUME_DIR="${RESUME_DIR:-}"
DEVICE="${DEVICE:-auto}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
# Use the full unified-memory pool if PyTorch's conservative MPS watermark gets
# in the way. Override to a smaller value if the desktop becomes memory starved.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

for required in \
  "$MODEL_DIR/llm.pt" \
  "$MODEL_DIR/CosyVoice-BlankEN" \
  "$CONFIG" \
  "$DATA_DIR/train.data.list" \
  "$DATA_DIR/dev.data.list"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$EXP_DIR" "$TB_DIR"
LOG_FILE="$EXP_DIR/console_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$ROOT_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Starting CosyVoice3 LoRA MPS training"
echo "repo=$ROOT_DIR"
echo "model=$MODEL_DIR"
echo "data=$DATA_DIR"
echo "exp=$EXP_DIR"
echo "tensorboard=$TB_DIR"
echo "log=$LOG_FILE"
echo "quality settings: fp32/no quantization, config=$CONFIG, lr=$LR, accum_grad=$ACCUM_GRAD, lora_r=$LORA_R"
echo "observability: train_log.jsonl, checkpoint_index.jsonl, TensorBoard, console log"
echo "checkpointing: every $SAVE_EVERY_STEPS optimizer step(s), eval every $EVAL_EVERY_STEPS step(s), best/latest/final maintained"

args=(
  local_voice_pipeline/train_lora_single_mps.py
  --model-dir "$MODEL_DIR"
  --config "$CONFIG"
  --train-data "$DATA_DIR/train.data.list"
  --cv-data "$DATA_DIR/dev.data.list"
  --out-dir "$EXP_DIR"
  --tensorboard-dir "$TB_DIR"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --accum-grad "$ACCUM_GRAD"
  --eval-every-steps "$EVAL_EVERY_STEPS"
  --save-every-steps "$SAVE_EVERY_STEPS"
  --max-eval-batches "$MAX_EVAL_BATCHES"
  --max-train-steps "$MAX_TRAIN_STEPS"
  --early-stop-patience "$EARLY_STOP_PATIENCE"
  --min-delta "$MIN_DELTA"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --lora-target-modules "$LORA_TARGET_MODULES"
)

if [[ -n "$RESUME_DIR" ]]; then
  args+=(--resume-dir "$RESUME_DIR")
fi

python "${args[@]}"
