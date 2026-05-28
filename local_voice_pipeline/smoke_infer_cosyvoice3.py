#!/usr/bin/env python3
"""Base CosyVoice3 inference smoke test.

This intentionally uses base llm.pt via AutoModel. It does not use llm.rl.pt.
Run only when the machine is free; it loads the model and can be slow on CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    parser.add_argument("--prompt-wav", default=str(ROOT_DIR / "asset/zero_shot_prompt.wav"))
    parser.add_argument("--prompt-text", default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")
    parser.add_argument("--text", default="You are a helpful assistant.<|endofprompt|>This is a short CosyVoice three smoke test.")
    parser.add_argument("--out", default=str(ROOT_DIR / "local_voice_pipeline/outputs/base_smoke.wav"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "llm.pt").exists():
        raise SystemExit(f"missing base checkpoint: {model_dir / 'llm.pt'}")
    if (model_dir / "llm.rl.pt").exists():
        print("note: llm.rl.pt exists in model dir but this smoke test loads base llm.pt via AutoModel", file=sys.stderr)

    cosyvoice = AutoModel(model_dir=str(model_dir), load_trt=False, load_vllm=False, fp16=False)
    outputs = list(cosyvoice.inference_zero_shot(
        args.text,
        args.prompt_text,
        args.prompt_wav,
        stream=False,
        text_frontend=True,
    ))
    if not outputs:
        raise SystemExit("no output generated")
    speech = torch.cat([item["tts_speech"] for item in outputs], dim=1)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), speech.cpu(), cosyvoice.sample_rate)
    print(f"wrote {out} sample_rate={cosyvoice.sample_rate} duration={speech.shape[1] / cosyvoice.sample_rate:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
