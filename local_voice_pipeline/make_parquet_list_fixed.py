#!/usr/bin/env python3
"""Reliable local replacement for tools/make_parquet_list.py.

The upstream helper currently schedules multiprocessing jobs without collecting
child exceptions; on this checkout the child can fail while still writing a
stale data.list. This script performs the same conversion synchronously and
verifies each parquet file exists before writing lists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def read_kv(path: Path, rest: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1 if rest else -1)
            if rest:
                if len(parts) == 2:
                    out[parts[0]] = parts[1]
            else:
                cols = line.split()
                if len(cols) >= 2:
                    out[cols[0]] = cols[1]
    return out


def optional_torch(path: Path):
    return torch.load(path, map_location="cpu") if path.exists() else None


def write_chunk(utts: list[str], parquet_file: Path, utt2wav, utt2text, utt2spk, utt2embedding, spk2embedding, utt2speech_token, utt2instruct) -> tuple[Path, Path, Path]:
    data = []
    for utt in tqdm(utts, desc=parquet_file.name):
        rec = {
            "utt": utt,
            "audio_data": Path(utt2wav[utt]).read_bytes(),
            "wav": utt2wav[utt],
            "text": utt2text[utt],
            "spk": utt2spk[utt],
        }
        if utt2embedding is not None:
            rec["utt_embedding"] = utt2embedding[utt]
        if spk2embedding is not None:
            rec["spk_embedding"] = spk2embedding[utt2spk[utt]]
        if utt2speech_token is not None:
            rec["speech_token"] = utt2speech_token[utt]
        if utt2instruct is not None:
            rec["instruct"] = utt2instruct[utt]
        data.append(rec)

    df = pd.DataFrame(data)
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_file)
    if not parquet_file.exists() or parquet_file.stat().st_size == 0:
        raise RuntimeError(f"failed to write parquet: {parquet_file}")

    idx = parquet_file.stem.split("_")[-1]
    utt2parquet_file = parquet_file.with_name(f"utt2parquet_{idx}.json")
    spk2parquet_file = parquet_file.with_name(f"spk2parquet_{idx}.json")
    utt2parquet_file.write_text(json.dumps({utt: str(parquet_file) for utt in utts}, ensure_ascii=False, indent=2), encoding="utf-8")
    spk2parquet_file.write_text(json.dumps({spk: str(parquet_file) for spk in sorted({utt2spk[utt] for utt in utts})}, ensure_ascii=False, indent=2), encoding="utf-8")
    return parquet_file, utt2parquet_file, spk2parquet_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_utts_per_parquet", type=int, default=1000)
    parser.add_argument("--num_processes", type=int, default=1, help="Accepted for CLI compatibility; conversion is synchronous.")
    parser.add_argument("--src_dir", required=True, type=Path)
    parser.add_argument("--des_dir", required=True, type=Path)
    parser.add_argument("--dpo", action="store_true", default=False, help="Not implemented in fixed local helper")
    args = parser.parse_args()
    if args.dpo:
        raise SystemExit("DPO parquet conversion is not implemented in local fixed helper")

    src_dir = args.src_dir.expanduser().resolve()
    des_dir = args.des_dir.expanduser().resolve()
    utt2wav = read_kv(src_dir / "wav.scp")
    utt2text = read_kv(src_dir / "text", rest=True)
    utt2spk = read_kv(src_dir / "utt2spk")
    utt2instruct = read_kv(src_dir / "instruct", rest=True) if (src_dir / "instruct").exists() else None
    utt2embedding = optional_torch(src_dir / "utt2embedding.pt")
    spk2embedding = optional_torch(src_dir / "spk2embedding.pt")
    utt2speech_token = optional_torch(src_dir / "utt2speech_token.pt")

    utts = list(utt2wav)
    if not utts:
        raise SystemExit(f"no utterances in {src_dir / 'wav.scp'}")
    missing_text = [u for u in utts if u not in utt2text]
    missing_spk = [u for u in utts if u not in utt2spk]
    if missing_text or missing_spk:
        raise SystemExit(f"missing text={missing_text[:5]} missing_spk={missing_spk[:5]}")

    parquet_files: list[Path] = []
    utt2parquet_files: list[Path] = []
    spk2parquet_files: list[Path] = []
    for i, start in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        chunk = utts[start:start + args.num_utts_per_parquet]
        parquet_file = des_dir / f"parquet_{i:09d}.tar"
        p, u, s = write_chunk(chunk, parquet_file, utt2wav, utt2text, utt2spk, utt2embedding, spk2embedding, utt2speech_token, utt2instruct)
        parquet_files.append(p)
        utt2parquet_files.append(u)
        spk2parquet_files.append(s)

    (des_dir / "data.list").write_text("".join(str(p) + "\n" for p in parquet_files), encoding="utf-8")
    (des_dir / "utt2data.list").write_text("".join(str(p) + "\n" for p in utt2parquet_files), encoding="utf-8")
    (des_dir / "spk2data.list").write_text("".join(str(p) + "\n" for p in spk2parquet_files), encoding="utf-8")
    print(f"wrote {len(parquet_files)} parquet file(s) to {des_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
