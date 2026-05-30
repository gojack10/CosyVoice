#!/usr/bin/env python3
"""Export the merged CosyVoice3 Qwen2 transformer checkpoint for Anemll.

This keeps the CosyVoice-specific speech embedding, embed assembly, speech-token
LM head, and sampler out of CoreML.  The exported directory is only the merged
HF Qwen2ForCausalLM backbone that Anemll's part-2 converter consumes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

import torch  # noqa: E402

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from local_voice_pipeline.synth_flow_ab import apply_lora_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"),
        help="CosyVoice3 model directory containing base llm.pt/cosyvoice3.yaml",
    )
    parser.add_argument(
        "--adapter-dir",
        default=str(ROOT_DIR / "local_voice_pipeline/exp/cosyvoice3/llm_lora/mps_single/best"),
        help="LoRA adapter directory to merge before export",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/cosyvoice3_ane/merged_qwen2",
        help="Output HF checkpoint directory for the merged Qwen2 backbone",
    )
    parser.add_argument("--no-merge-lora", action="store_true", help="Export PEFT wrapper unmerged; for debugging only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    for required in (model_dir / "llm.pt", model_dir / "cosyvoice3.yaml", adapter_dir / "lora_weights.pt"):
        if not required.exists():
            raise SystemExit(f"missing required path: {required}")

    cosyvoice = AutoModel(model_dir=str(model_dir), load_trt=False, load_vllm=False, fp16=False)
    apply_lora_adapter(cosyvoice, adapter_dir, merge_lora=not args.no_merge_lora)
    qwen = cosyvoice.model.llm.llm.model
    qwen.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    qwen.save_pretrained(str(out_dir), safe_serialization=True)

    lm = cosyvoice.model.llm
    meta = {
        "purpose": "CosyVoice3 ANE Stage-1 export: merged Qwen2 transformer backbone only",
        "model_dir": str(model_dir),
        "adapter_dir": str(adapter_dir),
        "merge_lora": not args.no_merge_lora,
        "qwen_class": qwen.__class__.__name__,
        "hidden_size": int(qwen.config.hidden_size),
        "num_hidden_layers": int(qwen.config.num_hidden_layers),
        "num_attention_heads": int(qwen.config.num_attention_heads),
        "num_key_value_heads": int(qwen.config.num_key_value_heads),
        "vocab_size": int(qwen.config.vocab_size),
        "speech_embedding_shape": list(lm.speech_embedding.weight.shape),
        "llm_decoder_shape": list(lm.llm_decoder.weight.shape),
        "speech_token_size": int(lm.speech_token_size),
        "sos": int(lm.sos),
        "task_id": int(lm.task_id),
        "stop_token_count": len(lm.stop_token_ids),
        "torch_version": torch.__version__,
    }
    (out_dir / "cosyvoice_ane_export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"EXPORT_OK out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
