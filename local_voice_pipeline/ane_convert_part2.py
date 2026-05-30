#!/usr/bin/env python3
"""Convert only the CosyVoice3 Qwen2 transformer (Anemll part-2) to CoreML.

Input is the merged HF Qwen2 backbone produced by ane_export_qwen2.py.  Output is
an FFN/transformer-only mlpackage: hidden_states in, hidden_states out.  The
CosyVoice speech embedding, text/speech embed assembly, llm_decoder, and sampler
remain host-side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anemll-dir", default="/tmp/anemll", help="Path to Anemll checkout")
    parser.add_argument("--model-dir", default="/tmp/cosyvoice3_ane/merged_qwen2", help="Merged HF Qwen2 export")
    parser.add_argument("--out-dir", default="/tmp/cosyvoice3_ane/coreml", help="Output directory for mlpackage")
    parser.add_argument("--prefix", default="cosyvoice3_qwen2", help="Output filename prefix")
    parser.add_argument("--context-length", type=int, default=1024, help="Fixed KV context/state length")
    parser.add_argument("--batch-size", type=int, default=64, help="Unused for part-2 infer; kept for converter API")
    parser.add_argument("--chunk", type=int, default=1, help="Single chunk is the intended Stage-1 path")
    parser.add_argument("--lut", type=str, default="none", help="LUT setting; default none/FP16 because LUT4 was ruled out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    anemll_dir = Path(args.anemll_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not (anemll_dir / "anemll/ane_converter/qwen2_5_converter.py").exists():
        raise SystemExit(f"missing Anemll qwen2_5_converter under {anemll_dir}")
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"missing exported HF config.json under {model_dir}")
    if args.chunk != 1:
        raise SystemExit("Stage-1 guardrail: use --chunk 1 so final RMSNorm is applied exactly once")

    sys.path.insert(0, str(anemll_dir))
    from anemll.ane_converter.qwen2_5_converter import parse_lut_arg, test_conversion  # noqa: WPS433

    lut_bits, per_channel = parse_lut_arg(args.lut)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "CONVERT_PART2_START "
        f"model_dir={model_dir} out_dir={out_dir} context={args.context_length} chunk={args.chunk} lut={lut_bits}"
    )
    test_conversion(
        model_path=str(model_dir),
        prefix=args.prefix,
        context_length=args.context_length,
        lut_bits=lut_bits,
        batch_size=args.batch_size,
        output_dir=str(out_dir),
        part="2",
        num_chunks=args.chunk,
        per_channel=per_channel,
    )

    expected = out_dir / f"{args.prefix}_FFN_chunk_01of01.mlpackage"
    meta = {
        "purpose": "CosyVoice3 ANE Stage-1 part-2 transformer conversion",
        "model_dir": str(model_dir),
        "mlpackage": str(expected),
        "context_length": args.context_length,
        "chunk": args.chunk,
        "lut_bits": lut_bits,
        "host_side": ["qwen embed_tokens/text embedding", "speech_embedding", "embed assembly", "llm_decoder", "sampling"],
        "post_final_rmsnorm_evidence": "Anemll qwen2_5_converter.convert_part_2 applies model.model.norm(out) when end_layer is None / final chunk; --chunk 1 enforces that path.",
    }
    (out_dir / "cosyvoice3_qwen2_part2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not expected.exists():
        raise SystemExit(f"conversion finished but expected mlpackage is missing: {expected}")
    print(json.dumps(meta, indent=2))
    print(f"CONVERT_PART2_OK mlpackage={expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
