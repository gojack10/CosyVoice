#!/usr/bin/env python3
"""Fast non-model-loading preflight for local CosyVoice3 adaptation readiness."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_MODEL_FILES = [
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "cosyvoice3.yaml",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/model.safetensors",
]
PY_MODULES = [
    "torch",
    "torchaudio",
    "gdown",
    "onnxruntime",
    "hyperpyyaml",
    "pandas",
    "pyarrow",
    "soundfile",
    "transformers",
    "peft",
    "whisper",
]


def module_status(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"ok": False, "error": "not importable"}
    try:
        mod = __import__(name)
        return {"ok": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:  # noqa: BLE001 - report import-specific failures
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def cmd_version(cmd: str) -> str | None:
    exe = shutil.which(cmd)
    if exe is None:
        return None
    try:
        proc = subprocess.run([exe, "-version"], text=True, capture_output=True, timeout=5)
        first = (proc.stdout or proc.stderr).splitlines()[0]
        return first
    except Exception:
        return exe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    parser.add_argument("--repo-dir", default=str(Path.home() / "projects/FunAudioLLM/CosyVoice"))
    parser.add_argument("--data-dir", default=str(Path.home() / "projects/FunAudioLLM/CosyVoice/local_voice_pipeline/data/cosyvoice3"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()

    report = {
        "python": sys.version.split()[0],
        "repo_dir": str(repo_dir),
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "ffmpeg": cmd_version("ffmpeg"),
        "ffprobe": shutil.which("ffprobe") is not None,
        "modules": {m: module_status(m) for m in PY_MODULES},
        "model_files": {f: (model_dir / f).exists() for f in REQUIRED_MODEL_FILES},
        "rl_checkpoint_present": (model_dir / "llm.rl.pt").exists(),
        "data_lists": {
            "train": (data_dir / "train.data.list").exists(),
            "dev": (data_dir / "dev.data.list").exists(),
        },
    }

    if report["modules"].get("torch", {}).get("ok"):
        import torch

        report["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    if report["modules"].get("onnxruntime", {}).get("ok"):
        import onnxruntime

        report["onnxruntime_providers"] = onnxruntime.get_available_providers()

    print(json.dumps(report, indent=2, sort_keys=True))

    hard_fail = []
    if not repo_dir.exists():
        hard_fail.append(f"repo dir missing: {repo_dir}")
    missing_model = [f for f, ok in report["model_files"].items() if not ok]
    if missing_model:
        hard_fail.append(f"missing model files: {missing_model}")
    missing_modules = [m for m, st in report["modules"].items() if not st.get("ok")]
    if missing_modules:
        hard_fail.append(f"missing/broken python modules: {missing_modules}")
    if report["ffmpeg"] is None or report["ffprobe"] is False:
        hard_fail.append("ffmpeg/ffprobe missing")

    if hard_fail:
        print("PREFLIGHT FAIL:", file=sys.stderr)
        for item in hard_fail:
            print(f"- {item}", file=sys.stderr)
        return 1
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
