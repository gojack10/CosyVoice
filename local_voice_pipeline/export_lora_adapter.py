#!/usr/bin/env python3
"""Export a CosyVoice3 LoRA training checkpoint into adapter-style files.

Output layout matches the Stanslab CosyVoice3 adapter clue:
  OUT/config.json
  OUT/lora_weights.pt
  OUT/speech_embedding.pt
  OUT/llm_decoder.pt

Input is a checkpoint produced by local_voice_pipeline/train_lora.py, typically
`epoch_...pt` under the LoRA exp directory. The script strips DDP/module and
CosyVoice parent prefixes so the adapter can be loaded by applying PEFT to the
base `llm.llm.model`, then strict-loading `speech_embedding` and `llm_decoder`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def clean_key(key: str) -> str:
    while key.startswith("module."):
        key = key.removeprefix("module.")
    return key


def tensor_state_only(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {clean_key(k): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Optional lora_train_metadata.json from the exp dir")
    parser.add_argument("--base-model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(checkpoint, map_location="cpu")
    state = tensor_state_only(raw)

    # Full CosyVoice3LM state has llm.model.<peft-key>. The inference adapter
    # applies PEFT directly to llm.llm.model, so lora_weights.pt should start at
    # base_model.model....
    lora = {}
    for k, v in state.items():
        if "lora_" not in k:
            continue
        if k.startswith("llm.model."):
            out_k = k.removeprefix("llm.model.")
        else:
            out_k = k
        lora[out_k] = v

    speech_embedding = {
        k.removeprefix("speech_embedding."): v
        for k, v in state.items()
        if k.startswith("speech_embedding.")
    }
    llm_decoder = {
        k.removeprefix("llm_decoder."): v
        for k, v in state.items()
        if k.startswith("llm_decoder.")
    }

    if not lora:
        raise SystemExit("no LoRA keys found; was this produced by train_lora.py?")
    if not speech_embedding:
        raise SystemExit("no speech_embedding keys found")
    if not llm_decoder:
        raise SystemExit("no llm_decoder keys found")

    torch.save(lora, out_dir / "lora_weights.pt")
    torch.save(speech_embedding, out_dir / "speech_embedding.pt")
    torch.save(llm_decoder, out_dir / "llm_decoder.pt")

    metadata = {}
    if args.metadata and args.metadata.exists():
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    config = {
        "base_model": args.base_model,
        "checkpoint": str(checkpoint),
        "format": "cosyvoice3_lora_adapter",
        "config": metadata,
        "files": ["lora_weights.pt", "speech_embedding.pt", "llm_decoder.pt"],
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"wrote adapter to {out_dir}")
    print(f"lora_tensors={len(lora)} speech_embedding_tensors={len(speech_embedding)} llm_decoder_tensors={len(llm_decoder)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
