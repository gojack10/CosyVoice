# Local CosyVoice3 Voice Adaptation Pipeline

Goal: make CosyVoice3 ready for data ingestion and first SFT training without depending on the final dataset yet.

This pipeline intentionally starts from the **base** CosyVoice3 checkpoint `llm.pt` in `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. It does **not** use `llm.rl.pt`, GRPO, or the unrelated CosyVoice2/yuekai GRPO checkpoints.

## Current local paths

- CosyVoice code: `~/projects/FunAudioLLM/CosyVoice`
- Base model target: `~/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- Local pipeline: `~/projects/FunAudioLLM/CosyVoice/local_voice_pipeline`

The official CosyVoice README uses `pretrained_models/Fun-CosyVoice3-0.5B`; these scripts use `MODEL_DIR` and default it to the Hugging Face repo path above.

## Official script map

- CosyVoice3 example inference: `example.py`, function `cosyvoice3_example()`
- Official CosyVoice3 training recipe: `examples/libritts/cosyvoice3/run.sh`
- Official CosyVoice3 config: `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`
- Local SFT config copy: `local_voice_pipeline/conf/cosyvoice3_sft.yaml`
- Local smoke config copy: `local_voice_pipeline/conf/cosyvoice3_smoke.yaml`
- Official trainer: `cosyvoice/bin/train.py`

Important findings:

- Official CosyVoice3 config uses `sample_rate: 24000`.
- Official LibriTTS prep writes Kaldi-style `wav.scp`, `text`, `utt2spk`, `spk2utt`, plus CosyVoice3 `instruct`.
- Official CosyVoice3 train command supports `--model llm`; use this first, not flow/hifigan/RL.
- Local data prep uses `local_voice_pipeline/make_parquet_list_fixed.py` instead of the upstream helper because the upstream multiprocessing helper can hide child-process failures while still writing `data.list`.
- The upstream repo does not currently expose a first-class LoRA training entrypoint. This local pipeline adds an experimental PEFT LoRA entrypoint in `local_voice_pipeline/train_lora.py`, using the adapter layout clue from `Stanslab/cosyvoice3-wolnelektury-v7`: LoRA on Qwen target modules plus trainable `speech_embedding` and `llm_decoder`.

## Data contract for the user-provided clips

Use one canonical JSONL manifest. Required fields are `utt_id`, `audio`, `text`, and `speaker`; recommended fields include `source`, `start`, `end`, and `split`.

```jsonl
{"utt_id":"video001_000123","audio":"clips/video001_000123.wav","text":"Exact words spoken in this clip.","speaker":"target","source":"video001.mp4","start":12.34,"end":18.90,"split":"train"}
```

Audio recommendations:

- Keep raw video/audio masters separately; do not destructively edit them.
- Training clips should be clean utterances, preferably 2-15 seconds; hard maximum 30 seconds because speech-token extraction skips/warns above that.
- Use mono 24 kHz PCM WAV for training copies. The prep script can convert to that from WAV/FLAC/M4A/MP4-readable sources via ffmpeg.
- Avoid clipped audio, background music, overlapping speakers, and long silence.
- If there are multiple speakers, only include the target speaker or assign correct speaker IDs and filter later.

Transcript recommendations:

- `text` should be the exact spoken words for the clip.
- Do not include the CosyVoice3 instruction prefix in `text`; the converter writes `instruct` as `You are a helpful assistant.<|endofprompt|>`.
- Preserve meaningful punctuation/case where useful; only whitespace is normalized by the converter.

## Prepare data once a manifest exists

From the CosyVoice checkout on macOS/Apple Silicon for local data prep and base inference:

```bash
cd ~/projects/FunAudioLLM/CosyVoice
./local_voice_pipeline/setup_macos_env.sh
source .venv/bin/activate
```

For CUDA/Linux training, start from the official `requirements.txt`, then add PEFT for LoRA:

```bash
pip install -r requirements.txt
pip install peft==0.14.0 python-multipart==0.0.20
```

Fast environment/model-file preflight, no model load:

```bash
python local_voice_pipeline/preflight.py \
  --model-dir ~/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

Validate only:

```bash
python local_voice_pipeline/validate_manifest.py \
  --manifest /path/to/manifest.jsonl \
  --audio-root /path/to/audio/root \
  --target-sample-rate 24000 \
  --target-channels 1
```

Convert to CosyVoice files, extract embeddings/tokens, and build parquet lists:

```bash
AUDIO_ROOT=/path/to/audio/root \
MODEL_DIR=~/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
SPEECH_TOKEN_PROVIDER=auto \
./local_voice_pipeline/prepare_cosyvoice3_data.sh /path/to/manifest.jsonl
```

Outputs:

- `local_voice_pipeline/data/cosyvoice3/train/{wav.scp,text,utt2spk,spk2utt,instruct}`
- `local_voice_pipeline/data/cosyvoice3/dev/{wav.scp,text,utt2spk,spk2utt,instruct}`
- `local_voice_pipeline/data/cosyvoice3/train/parquet/data.list`
- `local_voice_pipeline/data/cosyvoice3/dev/parquet/data.list`
- `local_voice_pipeline/data/cosyvoice3/train.data.list`
- `local_voice_pipeline/data/cosyvoice3/dev.data.list`

## Smoke tests to run only when the machine is free

Base inference smoke, no training:

```bash
cd ~/projects/FunAudioLLM/CosyVoice
source .venv/bin/activate
python local_voice_pipeline/smoke_infer_cosyvoice3.py \
  --model-dir ~/projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --out local_voice_pipeline/outputs/base_smoke.wav
```

Tiny LLM SFT/data-loader smoke, requires CUDA/NVIDIA because the official trainer uses torch distributed CUDA:

```bash
cd ~/projects/FunAudioLLM/CosyVoice
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 ./local_voice_pipeline/smoke_train_llm_sft.sh
```

Local single-process LoRA smoke, can run on Apple Silicon MPS/CPU after data prep:

```bash
cd ~/projects/FunAudioLLM/CosyVoice
source .venv/bin/activate
PYTORCH_ENABLE_MPS_FALLBACK=1 python local_voice_pipeline/smoke_lora_single.py \
  --train-data local_voice_pipeline/data/cosyvoice3/train.data.list \
  --out-dir local_voice_pipeline/outputs/lora_smoke_adapter
```

Tiny LoRA/DDP data-loader smoke, CUDA/NVIDIA only:

```bash
cd ~/projects/FunAudioLLM/CosyVoice
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 ./local_voice_pipeline/smoke_train_llm_lora.sh
```

First real full LLM SFT after smoke passes:

```bash
cd ~/projects/FunAudioLLM/CosyVoice
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 ./local_voice_pipeline/train_llm_sft.sh
```

First real LoRA/adapter run after smoke passes:

```bash
cd ~/projects/FunAudioLLM/CosyVoice
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 ./local_voice_pipeline/train_llm_lora.sh
```

Export a LoRA checkpoint into adapter files after a LoRA run:

```bash
python local_voice_pipeline/export_lora_adapter.py \
  --checkpoint local_voice_pipeline/exp/cosyvoice3/llm_lora/torch_ddp/epoch_0_whole.pt \
  --metadata local_voice_pipeline/exp/cosyvoice3/llm_lora/torch_ddp/lora_train_metadata.json \
  --out-dir local_voice_pipeline/outputs/adapter_step_000000
```

## Stop lines

- Do not use `llm.rl.pt` for first voice adaptation.
- Do not run RL/GRPO until there is an ASR/reward/eval loop and rollback criteria.
- Do not use `yuekai/*cosyvoice-llm-grpo-aishell3` as the CosyVoice3 base; those are CosyVoice2/AISHELL3 experiments.
- Do not transcribe/process the full video set before a small pilot manifest validates and a smoke run passes.
