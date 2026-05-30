#!/usr/bin/env python3
"""Split vega.wav into per-segment clips and write the JSONL manifest."""
import json, subprocess, sys
from pathlib import Path

TRANSCRIPT = Path("~/Downloads/vega-transcription-20260528T221445.json").expanduser()
WAV = Path("~/projects/FunAudioLLM/CosyVoice/local_voice_pipeline/data/raw_vega/vega.wav").expanduser()
OUT_DIR = Path("~/projects/FunAudioLLM/CosyVoice/local_voice_pipeline/data/raw_vega/clips").expanduser()

with open(TRANSCRIPT) as f:
    data = json.load(f)

OUT_DIR.mkdir(parents=True, exist_ok=True)
manifest_lines = []
skipped = 0

for i, seg in enumerate(data["segments"]):
    idx = f"{i+1:04d}"
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()
    duration = end - start

    if duration < 1.0:
        skipped += 1
        continue

    clip_path = OUT_DIR / f"vega_{idx}.wav"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-to", str(end),
        "-i", str(WAV),
        "-vn", "-ac", "1", "-ar", "24000",
        "-sample_fmt", "s16", str(clip_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"ERROR splitting {idx}: {proc.stderr}", file=sys.stderr)
        continue

    manifest_lines.append(json.dumps({
        "utt_id": f"vega_{idx}",
        "audio": f"clips/vega_{idx}.wav",
        "text": text,
        "speaker": "vega",
        "source": "vega.wav",
        "start": start,
        "end": end,
        "split": "train",
    }))

# Put ~10% into dev for validation
import random
records = [json.loads(line) for line in manifest_lines]
utt_ids = [int(r["utt_id"].split("_")[1]) for r in records]
random.seed(42)
dev_ids = set(random.sample(utt_ids, max(1, len(utt_ids) // 10)))

final_lines = []
for rec in records:
    utt_num = int(rec["utt_id"].split("_")[1])
    if utt_num in dev_ids:
        rec["split"] = "dev"
    final_lines.append(json.dumps(rec))

manifest_path = OUT_DIR.parent / "manifest.jsonl"
with open(manifest_path, "w") as f:
    f.write("\n".join(final_lines) + "\n")

print(f"Wrote {len(final_lines)} clips to {OUT_DIR}/")
print(f"Manifest: {manifest_path}")
print(f"  train: {sum(1 for l in final_lines if json.loads(l)['split']=='train')}")
print(f"  dev:   {sum(1 for l in final_lines if json.loads(l)['split']=='dev')}")
print(f"Skipped {skipped} sub-1s segments")