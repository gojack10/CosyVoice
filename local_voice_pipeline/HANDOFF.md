# CosyVoice3 Adaptation Handoff

Status: ready for user data ingestion and first SFT/LoRA smoke/launch commands.

## What is set up

- CosyVoice repo: `~/projects/FunAudioLLM/CosyVoice`
- Base model files: `~/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- Convenience symlink: `pretrained_models/Fun-CosyVoice3-0.5B -> ~/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- Local pipeline: `local_voice_pipeline/`
- Python env: `~/projects/FunAudioLLM/CosyVoice/.venv` using Python 3.10.20

Base model download intentionally skipped `llm.rl.pt`; first adaptation starts from `llm.pt`.

## Verified evidence

- `python local_voice_pipeline/preflight.py --model-dir ~/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512` returned `PREFLIGHT OK`.
  - Required model files present: `llm.pt`, `flow.pt`, `hift.pt`, `cosyvoice3.yaml`, `campplus.onnx`, `speech_tokenizer_v3.onnx`, `CosyVoice-BlankEN/model.safetensors`.
  - Python modules present: torch, torchaudio, onnxruntime, hyperpyyaml, pandas, pyarrow, soundfile, transformers, peft, whisper, gdown.
  - Local hardware: CUDA unavailable, MPS available.
- Synthetic manifest/data-prep smoke succeeded.
  - `validate_manifest.py` accepted a two-record manifest.
  - `prepare_cosyvoice3_data.sh` extracted embeddings/tokens and wrote real parquet files.
  - Verified parquet columns include `audio_data`, `text`, `spk`, `instruct`, `utt_embedding`, `spk_embedding`, `speech_token`.
- Base inference smoke succeeded.
  - Command wrote `local_voice_pipeline/outputs/base_smoke.wav`.
  - Output verified as mono 24 kHz, duration 1.44s.
- Local single-process LoRA smoke succeeded on MPS.
  - One synthetic batch: `step=1 loss=2.620512 acc=0.588235`.
  - Adapter-style files were written: `config.json`, `lora_weights.pt`, `speech_embedding.pt`, `llm_decoder.pt`.

## User data contract

Create a JSONL manifest with one record per clean clip:

```jsonl
{"utt_id":"video001_000123","audio":"clips/video001_000123.wav","text":"Exact words spoken in this clip.","speaker":"target","source":"video001.mp4","start":12.34,"end":18.90,"split":"train"}
```

Required fields: `utt_id`, `audio`, `text`, `speaker`.
Recommended: `source`, `start`, `end`, `split`.

Audio target: clean mono 24 kHz PCM WAV training copies. Keep raw videos/masters separately. Prefer 2-15s clips, hard max 30s.

## When user returns with data

1. Put clips somewhere stable, e.g. `local_voice_pipeline/data/clips/` or another path.
2. Create a manifest JSONL.
3. Validate:

```bash
cd ~/projects/FunAudioLLM/CosyVoice
source .venv/bin/activate
python local_voice_pipeline/validate_manifest.py \
  --manifest /path/to/manifest.jsonl \
  --audio-root /path/to/audio/root \
  --target-sample-rate 24000 \
  --target-channels 1
```

4. Prepare CosyVoice data:

```bash
AUDIO_ROOT=/path/to/audio/root \
MODEL_DIR=~/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
SPEECH_TOKEN_PROVIDER=auto \
./local_voice_pipeline/prepare_cosyvoice3_data.sh /path/to/manifest.jsonl
```

5. Optional local LoRA smoke on Apple Silicon/MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python local_voice_pipeline/smoke_lora_single.py \
  --train-data local_voice_pipeline/data/cosyvoice3/train.data.list \
  --out-dir local_voice_pipeline/outputs/lora_smoke_adapter
```

6. CUDA/Linux first training commands:

```bash
CUDA_VISIBLE_DEVICES=0 ./local_voice_pipeline/smoke_train_llm_lora.sh
CUDA_VISIBLE_DEVICES=0 ./local_voice_pipeline/train_llm_lora.sh
```

or full SFT instead of LoRA:

```bash
CUDA_VISIBLE_DEVICES=0 ./local_voice_pipeline/smoke_train_llm_sft.sh
CUDA_VISIBLE_DEVICES=0 ./local_voice_pipeline/train_llm_sft.sh
```

## Stop lines

- Do not use `llm.rl.pt` for first voice adaptation.
- Do not start RL/GRPO until ASR/reward/eval and rollback criteria exist.
- Do not use `yuekai/*cosyvoice-llm-grpo-aishell3`; those are CosyVoice2/AISHELL3 experiments.
