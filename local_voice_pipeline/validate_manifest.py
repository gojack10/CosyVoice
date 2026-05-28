#!/usr/bin/env python3
"""Validate canonical CosyVoice3 voice-cloning/fine-tuning JSONL manifests.

Expected JSONL record:
  {"utt_id":"...","audio":"clips/foo.wav","text":"...","speaker":"target",
   "source":"video001.mp4","start":12.34,"end":18.90,"split":"train"}

Only utt_id/audio/text/speaker are required. Paths are resolved relative to
--audio-root first, then relative to the manifest file directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ALLOWED_SPLITS = {"train", "dev", "valid", "validation", "test"}


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def resolve_audio(audio: str, manifest: Path, audio_root: Path | None) -> Path:
    p = Path(audio).expanduser()
    if p.is_absolute():
        return p
    candidates = []
    if audio_root is not None:
        candidates.append((audio_root / p).resolve())
    candidates.append((manifest.parent / p).resolve())
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    return json.loads(proc.stdout)


def audio_info(meta: dict[str, Any]) -> tuple[float | None, int | None, int | None, str | None]:
    streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        return None, None, None, None
    s = streams[0]
    duration = s.get("duration") or meta.get("format", {}).get("duration")
    try:
        duration_f = float(duration) if duration is not None else None
    except ValueError:
        duration_f = None
    try:
        sr = int(s["sample_rate"]) if s.get("sample_rate") else None
    except ValueError:
        sr = None
    try:
        channels = int(s["channels"]) if s.get("channels") else None
    except ValueError:
        channels = None
    return duration_f, sr, channels, s.get("codec_name")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="Base directory for relative audio paths. Defaults to manifest directory.")
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--target-channels", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0,
                        help="CosyVoice speech-token extraction warns/skips >30s clips.")
    parser.add_argument("--strict-format", action="store_true",
                        help="Treat sample-rate/channel mismatches as errors instead of warnings.")
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe not found; install ffmpeg first")

    manifest = args.manifest.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve() if args.audio_root else None
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    durations: list[float] = []
    missing_by_speaker: defaultdict[str, int] = defaultdict(int)

    for line_no, rec in load_jsonl(manifest):
        prefix = f"{manifest}:{line_no}"
        for field in ("utt_id", "audio", "text", "speaker"):
            if not str(rec.get(field, "")).strip():
                errors.append(f"{prefix}: missing/empty required field {field!r}")
        utt_id = str(rec.get("utt_id", ""))
        if utt_id in seen:
            errors.append(f"{prefix}: duplicate utt_id {utt_id!r}")
        seen.add(utt_id)
        if any(ch.isspace() for ch in utt_id):
            errors.append(f"{prefix}: utt_id must not contain whitespace: {utt_id!r}")
        split = str(rec.get("split", "train")).lower()
        if split and split not in ALLOWED_SPLITS:
            errors.append(f"{prefix}: split {split!r} not in {sorted(ALLOWED_SPLITS)}")
        split_counts["dev" if split in {"valid", "validation"} else split or "train"] += 1
        speaker = str(rec.get("speaker", "")) or "<missing>"
        speaker_counts[speaker] += 1

        audio_value = str(rec.get("audio", ""))
        if not audio_value:
            missing_by_speaker[speaker] += 1
            continue
        audio_path = resolve_audio(audio_value, manifest, audio_root)
        if not audio_path.exists():
            errors.append(f"{prefix}: audio not found: {audio_path}")
            missing_by_speaker[speaker] += 1
            continue
        try:
            duration, sr, channels, codec = audio_info(ffprobe(audio_path))
        except Exception as exc:  # noqa: BLE001 - surface file-specific probe failures
            errors.append(f"{prefix}: ffprobe failed for {audio_path}: {exc}")
            continue
        if duration is not None:
            durations.append(duration)
            if duration < args.min_seconds:
                errors.append(f"{prefix}: duration {duration:.2f}s < {args.min_seconds:.2f}s")
            if duration > args.max_seconds:
                errors.append(f"{prefix}: duration {duration:.2f}s > {args.max_seconds:.2f}s")
        if sr != args.target_sample_rate:
            msg = f"{prefix}: sample_rate {sr} != target {args.target_sample_rate} ({audio_path})"
            (errors if args.strict_format else warnings).append(msg)
        if channels != args.target_channels:
            msg = f"{prefix}: channels {channels} != target {args.target_channels} ({audio_path})"
            (errors if args.strict_format else warnings).append(msg)
        if codec and codec not in {"pcm_s16le", "pcm_s24le", "flac"}:
            warnings.append(f"{prefix}: codec {codec!r}; prefer PCM WAV or FLAC masters")

    print(f"records={len(seen)}")
    print("splits=" + json.dumps(dict(split_counts), sort_keys=True))
    print("speakers=" + json.dumps(dict(speaker_counts.most_common()), ensure_ascii=False))
    if durations:
        print(f"duration_seconds=min:{min(durations):.2f} median:{sorted(durations)[len(durations)//2]:.2f} max:{max(durations):.2f} total:{sum(durations):.2f}")
    for w in warnings:
        print("WARNING " + w, file=sys.stderr)
    for e in errors:
        print("ERROR " + e, file=sys.stderr)
    if errors:
        return 1
    print("manifest validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
