#!/usr/bin/env bash
set -euo pipefail

# Convert/validate a canonical JSONL manifest and build CosyVoice3 parquet lists.
# This does not train or load the LLM. It does require Python deps plus the
# campplus/speech-tokenizer ONNX files from the base model directory.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$ROOT_DIR/local_voice_pipeline"
MODEL_DIR="${MODEL_DIR:-$HOME/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
MANIFEST="${1:-${MANIFEST:-}}"
OUT_DIR="${OUT_DIR:-$PIPELINE_DIR/data/cosyvoice3}"
AUDIO_ROOT="${AUDIO_ROOT:-}"
SPLITS="${SPLITS:-train dev}"
SPEECH_TOKEN_PROVIDER="${SPEECH_TOKEN_PROVIDER:-auto}" # auto|cuda|cpu
NUM_THREAD="${NUM_THREAD:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

if [[ -z "$MANIFEST" ]]; then
  echo "usage: $0 path/to/manifest.jsonl" >&2
  echo "optional env: MODEL_DIR OUT_DIR AUDIO_ROOT SPEECH_TOKEN_PROVIDER NUM_THREAD NUM_PROCESSES" >&2
  exit 2
fi

if [[ ! -f "$MODEL_DIR/campplus.onnx" ]]; then
  echo "missing $MODEL_DIR/campplus.onnx; download base model first" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/speech_tokenizer_v3.onnx" ]]; then
  echo "missing $MODEL_DIR/speech_tokenizer_v3.onnx; download base model first" >&2
  exit 1
fi

VALIDATE_ARGS=(--manifest "$MANIFEST" --target-sample-rate 24000 --target-channels 1)
CONVERT_ARGS=(--manifest "$MANIFEST" --out-dir "$OUT_DIR" --convert-audio --sample-rate 24000)
if [[ -n "$AUDIO_ROOT" ]]; then
  VALIDATE_ARGS+=(--audio-root "$AUDIO_ROOT")
  CONVERT_ARGS+=(--audio-root "$AUDIO_ROOT")
fi

python "$PIPELINE_DIR/validate_manifest.py" "${VALIDATE_ARGS[@]}"
python "$PIPELINE_DIR/jsonl_to_kaldi.py" "${CONVERT_ARGS[@]}"

for split in $SPLITS; do
  split_dir="$OUT_DIR/$split"
  [[ -f "$split_dir/wav.scp" ]] || continue
  echo "Preparing split: $split_dir"
  python "$ROOT_DIR/tools/extract_embedding.py" \
    --dir "$split_dir" \
    --onnx_path "$MODEL_DIR/campplus.onnx" \
    --num_thread "$NUM_THREAD"
  python "$PIPELINE_DIR/extract_speech_token_fallback.py" \
    --dir "$split_dir" \
    --onnx_path "$MODEL_DIR/speech_tokenizer_v3.onnx" \
    --provider "$SPEECH_TOKEN_PROVIDER" \
    --num_thread "$NUM_THREAD"
  mkdir -p "$split_dir/parquet"
  python "$PIPELINE_DIR/make_parquet_list_fixed.py" \
    --num_utts_per_parquet 1000 \
    --num_processes "$NUM_PROCESSES" \
    --src_dir "$split_dir" \
    --des_dir "$split_dir/parquet"
done

if [[ -f "$OUT_DIR/train/parquet/data.list" ]]; then
  cp "$OUT_DIR/train/parquet/data.list" "$OUT_DIR/train.data.list"
fi
if [[ -f "$OUT_DIR/dev/parquet/data.list" ]]; then
  cp "$OUT_DIR/dev/parquet/data.list" "$OUT_DIR/dev.data.list"
fi

echo "CosyVoice3 data prepared under $OUT_DIR"
