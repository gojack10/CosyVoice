#!/usr/bin/env bash
set -euo pipefail

# Download the base CosyVoice3 model files needed for inference/data prep/SFT.
# By default this skips llm.rl.pt because first voice adaptation starts from llm.pt.
# It also skips the large TensorRT export ONNX unless INCLUDE_TRT_ONNX=1.

MODEL_DIR="${MODEL_DIR:-$HOME/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
INCLUDE_RL="${INCLUDE_RL:-0}"
INCLUDE_TRT_ONNX="${INCLUDE_TRT_ONNX:-0}"

mkdir -p "$MODEL_DIR"
args=(FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir "$MODEL_DIR" --max-workers "${MAX_WORKERS:-4}")
if [[ "$INCLUDE_RL" != "1" ]]; then
  args+=(--exclude llm.rl.pt)
fi
if [[ "$INCLUDE_TRT_ONNX" != "1" ]]; then
  args+=(--exclude flow.decoder.estimator.fp32.onnx)
fi

hf download "${args[@]}"

echo "Base CosyVoice3 model files are under $MODEL_DIR"
echo "Expected base checkpoint: $MODEL_DIR/llm.pt"
