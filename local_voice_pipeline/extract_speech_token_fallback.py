#!/usr/bin/env python3
"""CosyVoice speech-token extraction with CUDA-or-CPU ONNX provider fallback.

The upstream tools/extract_speech_token.py hard-codes CUDAExecutionProvider. This
variant keeps the same output contract (utt2speech_token.pt) but can run on a
CPU-only machine for small pilots. It is expected to be slow on CPU.
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torchaudio
import whisper
from tqdm import tqdm


def load_wav_map(data_dir: Path) -> dict[str, str]:
    utt2wav: dict[str, str] = {}
    with (data_dir / "wav.scp").open("r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split(maxsplit=1)
            if len(cols) == 2:
                utt2wav[cols[0]] = cols[1]
    return utt2wav


def provider_list(prefer: str) -> list[str]:
    available = set(onnxruntime.get_available_providers())
    if prefer == "cpu":
        return ["CPUExecutionProvider"]
    if prefer == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise SystemExit(f"CUDAExecutionProvider unavailable; available={sorted(available)}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--onnx_path", required=True, type=Path)
    parser.add_argument("--num_thread", type=int, default=4)
    parser.add_argument("--provider", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    utt2wav = load_wav_map(args.dir)
    if not utt2wav:
        raise SystemExit(f"no utterances found in {args.dir / 'wav.scp'}")

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = provider_list(args.provider)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("using ONNX providers: %s", providers)
    session = onnxruntime.InferenceSession(str(args.onnx_path), sess_options=option, providers=providers)

    def single_job(utt: str) -> tuple[str, list[int]]:
        audio, sample_rate = torchaudio.load(utt2wav[utt], backend="soundfile")
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[1] / 16000 > 30:
            logging.warning("%s longer than 30s; writing empty speech token", utt)
            return utt, []
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        speech_token = session.run(
            None,
            {
                session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0].flatten().tolist()
        return utt, speech_token

    utt2speech_token: dict[str, list[int]] = {}
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        futures = [executor.submit(single_job, utt) for utt in utt2wav]
        for future in tqdm(as_completed(futures), total=len(futures)):
            utt, speech_token = future.result()
            utt2speech_token[utt] = speech_token

    torch.save(utt2speech_token, args.dir / "utt2speech_token.pt")
    logging.info("wrote %s", args.dir / "utt2speech_token.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
