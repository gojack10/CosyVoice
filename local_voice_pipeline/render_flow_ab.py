#!/usr/bin/env python3
"""Render held-out lines for a base-vs-finetuned FLOW A/B audition.

Loads base CosyVoice3 via AutoModel (the proven smoke recipe), optionally swaps a
finetuned flow checkpoint into cosyvoice.model.flow, and renders zero-shot from a
VEGA prompt clip. Keeps the LLM constant so the audible delta is the flow.

Compare, per held-out utt:
  outputs/flow_ab/base__<utt>.wav   vs   outputs/flow_ab/<ft-tag>__<utt>.wav
and both against the real clip data/cosyvoice3/audio_24k/<utt>.wav.
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

INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
PROMPT_DIR = ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/audio_24k"
DEFAULT_PROMPT_WAV = str(PROMPT_DIR / "vega_0002.wav")
DEFAULT_PROMPT_TEXT = (
    "After running diagnostics on the Praetor suit, it appears that I can activate "
    "optional challenges that, when completed, will assist in upgrading your arsenal "
    "at an accelerated pace."
)


def with_instruct(text: str) -> str:
    return text if "<|endofprompt|>" in text else INSTRUCT + text


def read_dev_texts(dev_text: Path, n: int) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for line in dev_text.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        utt, _, text = line.partition(" ")
        if text:
            items.append((utt, text))
    # Longer lines clone far better in zero-shot than 3-5 word fragments.
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    return items[:n]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    p.add_argument("--flow-ckpt", default="", help="finetuned flow .pt to load into cosyvoice.model.flow; empty = base")
    p.add_argument("--prompt-wav", default=DEFAULT_PROMPT_WAV)
    p.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    p.add_argument("--dev-text", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/dev/text"))
    p.add_argument("--texts", default="", help="optional ';'-separated lines to render instead of dev/text")
    p.add_argument("--num-texts", type=int, default=3)
    p.add_argument("--out-dir", default=str(ROOT_DIR / "local_voice_pipeline/outputs/flow_ab"))
    p.add_argument("--tag", default="base")
    p.add_argument("--flow-steps", type=int, default=0, help="0 = model default; else set flow._inference_timesteps")
    p.add_argument("--flow-cfg-rate", type=float, default=-1.0, help="<0 = model default; else set flow.decoder._inference_cfg_rate")
    args = p.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not (model_dir / "flow.pt").exists():
        raise SystemExit(f"missing base flow: {model_dir / 'flow.pt'}")

    cosyvoice = AutoModel(model_dir=str(model_dir), load_trt=False, load_vllm=False, fp16=False)

    if args.flow_ckpt:
        ckpt = Path(args.flow_ckpt).expanduser().resolve()
        if not ckpt.exists():
            raise SystemExit(f"missing flow ckpt: {ckpt}")
        state = {k: v for k, v in torch.load(ckpt, map_location="cpu").items() if isinstance(v, torch.Tensor)}
        missing, unexpected = cosyvoice.model.flow.load_state_dict(state, strict=False)
        print(f"loaded finetuned flow {ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if args.flow_steps > 0:
        cosyvoice.model.flow._inference_timesteps = args.flow_steps
    if args.flow_cfg_rate >= 0:
        cosyvoice.model.flow.decoder._inference_cfg_rate = args.flow_cfg_rate

    if args.texts.strip():
        items = [(f"line{idx:02d}", t.strip()) for idx, t in enumerate(args.texts.split(";")) if t.strip()]
    else:
        items = read_dev_texts(Path(args.dev_text).expanduser().resolve(), args.num_texts)
    if not items:
        raise SystemExit("no texts to render")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = with_instruct(args.prompt_text)
    print(f"tag={args.tag} prompt_wav={args.prompt_wav} flow_ckpt={args.flow_ckpt or '(base)'}", flush=True)

    for utt, text in items:
        tts_text = with_instruct(text)
        outputs = list(cosyvoice.inference_zero_shot(tts_text, prompt_text, args.prompt_wav, stream=False, text_frontend=True))
        if not outputs:
            print(f"WARN no output for {utt}", flush=True)
            continue
        speech = torch.cat([o["tts_speech"] for o in outputs], dim=1)
        out_path = out_dir / f"{args.tag}__{utt}.wav"
        torchaudio.save(str(out_path), speech.cpu(), cosyvoice.sample_rate)
        print(f"wrote {out_path} dur={speech.shape[1] / cosyvoice.sample_rate:.2f}s text={text[:70]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
