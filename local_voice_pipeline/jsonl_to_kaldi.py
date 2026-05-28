#!/usr/bin/env python3
"""Convert canonical JSONL into the files CosyVoice expects.

Output layout:
  OUT/train/{wav.scp,text,utt2spk,spk2utt,instruct}
  OUT/dev/{wav.scp,text,utt2spk,spk2utt,instruct}

The text file contains only the exact transcript. CosyVoice3 instruction prompt
is written separately to the instruct file, matching official LibriTTS prep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

ALLOWED_SPLITS = {"train", "dev", "valid", "validation", "test"}
DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
UTT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_line_no"] = line_no
            yield rec


def resolve_audio(audio: str, manifest: Path, audio_root: Path | None) -> Path:
    p = Path(audio).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidates = []
    if audio_root is not None:
        candidates.append((audio_root / p).resolve())
    candidates.append((manifest.parent / p).resolve())
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def clean_text(text: str) -> str:
    # Keep punctuation/case; only collapse whitespace because Kaldi-style text is one line per utt.
    return " ".join(text.replace("\u00a0", " ").split())


def normalize_split(value: str | None) -> str | None:
    if not value:
        return None
    value = value.lower()
    if value in {"valid", "validation"}:
        return "dev"
    if value not in ALLOWED_SPLITS:
        raise ValueError(f"unknown split {value!r}")
    return value


def hash_to_dev(utt_id: str, dev_fraction: float) -> bool:
    h = hashlib.sha1(utt_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return bucket < dev_fraction


def maybe_convert_audio(src: Path, dst: Path, sample_rate: int) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-sample_fmt", "s16", str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst.resolve()


def write_split(records: list[dict], out_dir: Path, instruct: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    spk2utt: defaultdict[str, list[str]] = defaultdict(list)
    for rec in records:
        spk2utt[rec["speaker"]].append(rec["utt_id"])

    with (out_dir / "wav.scp").open("w", encoding="utf-8") as wav_f, \
            (out_dir / "text").open("w", encoding="utf-8") as text_f, \
            (out_dir / "utt2spk").open("w", encoding="utf-8") as utt2spk_f, \
            (out_dir / "instruct").open("w", encoding="utf-8") as instruct_f:
        for rec in sorted(records, key=lambda r: r["utt_id"]):
            wav_f.write(f"{rec['utt_id']} {rec['audio_path']}\n")
            text_f.write(f"{rec['utt_id']} {rec['text']}\n")
            utt2spk_f.write(f"{rec['utt_id']} {rec['speaker']}\n")
            instruct_f.write(f"{rec['utt_id']} {rec.get('instruct') or instruct}\n")

    with (out_dir / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--convert-audio", action="store_true",
                        help="Write 24k mono PCM WAV copies under OUT/audio_24k and point wav.scp there.")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--single-speaker-id", default=None,
                        help="Override all record speaker IDs; useful for target-only adaptation.")
    args = parser.parse_args()

    if args.convert_audio and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found; needed for --convert-audio")

    manifest = args.manifest.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve() if args.audio_root else None
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen: set[str] = set()
    for rec in load_jsonl(manifest):
        utt_id = str(rec.get("utt_id", "")).strip()
        if not UTT_RE.match(utt_id):
            raise SystemExit(f"line {rec['_line_no']}: invalid utt_id {utt_id!r}; use no spaces and stable ASCII IDs")
        if utt_id in seen:
            raise SystemExit(f"line {rec['_line_no']}: duplicate utt_id {utt_id!r}")
        seen.add(utt_id)
        text = clean_text(str(rec.get("text", "")))
        if not text:
            raise SystemExit(f"line {rec['_line_no']}: empty text")
        speaker = args.single_speaker_id or str(rec.get("speaker", "target")).strip() or "target"
        if any(ch.isspace() for ch in speaker):
            raise SystemExit(f"line {rec['_line_no']}: speaker ID must not contain whitespace: {speaker!r}")
        src_audio = resolve_audio(str(rec.get("audio", "")), manifest, audio_root)
        if not src_audio.exists():
            raise SystemExit(f"line {rec['_line_no']}: audio not found: {src_audio}")
        if args.convert_audio:
            suffix = ".wav"
            audio_path = maybe_convert_audio(src_audio, out_dir / "audio_24k" / f"{utt_id}{suffix}", args.sample_rate)
        else:
            audio_path = src_audio
        split = normalize_split(rec.get("split"))
        records.append({
            "utt_id": utt_id,
            "audio_path": str(audio_path),
            "text": text,
            "speaker": speaker,
            "split": split,
            "instruct": rec.get("instruct"),
        })

    if not records:
        raise SystemExit("manifest has no records")

    # Assign missing splits deterministically. For very small pilots, force one dev item when possible.
    random.Random(args.seed).shuffle(records)
    missing = [r for r in records if r["split"] is None]
    for r in missing:
        r["split"] = "dev" if hash_to_dev(r["utt_id"], args.dev_fraction) else "train"
    if len(records) > 1 and not any(r["split"] == "dev" for r in records):
        records[-1]["split"] = "dev"
    if len(records) == 1 and records[0]["split"] == "dev":
        records[0]["split"] = "train"

    by_split: defaultdict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    for split in ("train", "dev", "test"):
        if by_split.get(split):
            write_split(by_split[split], out_dir / split, args.instruct)
            print(f"wrote {len(by_split[split])} records to {out_dir / split}")
    if not by_split.get("dev"):
        print("WARNING: no dev records; training command needs a dev.data.list", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
