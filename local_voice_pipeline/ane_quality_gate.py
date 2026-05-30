#!/usr/bin/env python3
"""Quality gate for the CosyVoice3 ANE LLM spike.

No audio is synthesized here.  The gate teacher-forces the accepted MLX-generated
speech-token sequence through:
  1. the existing MLX Qwen speech-token path (reference), and
  2. the Anemll/CoreML part-2 transformer with host-side embeddings + llm_decoder.

It reports per-step logit KL and top-k overlap.  If the thresholds fail, do not
wire the ANE path into audio.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from local_voice_pipeline.mlx_qwen_backend import MlxQwenSpeechLM  # noqa: E402
from local_voice_pipeline.synth_flow_ab import apply_lora_adapter, _embed_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    parser.add_argument("--adapter-dir", default=str(ROOT_DIR / "local_voice_pipeline/exp/cosyvoice3/llm_lora/mps_single/best"))
    parser.add_argument(
        "--tokens-json",
        default=str(
            ROOT_DIR
            / "local_voice_pipeline/outputs/regen_prefix448_compare/regen_prefix448_compare_20260529_122951/regen_mlx_prefix_448_flow10.tokens.json"
        ),
        help="Accepted MLX token dump used as the teacher-forced sequence",
    )
    parser.add_argument("--chunk-index", type=int, default=0, help="0-based token dump chunk to gate")
    parser.add_argument("--mlpackage", default="/tmp/cosyvoice3_ane/coreml/cosyvoice3_qwen2_FFN_chunk_01of01.mlpackage")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = all generated tokens for selected chunk")
    parser.add_argument("--mlx-dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument(
        "--compute-units",
        default="cpu_and_ne",
        choices=("all", "cpu_and_ne", "cpu_only", "cpu_and_gpu"),
        help="CoreML compute units for the part-2 model",
    )
    parser.add_argument("--mean-kl-threshold", type=float, default=0.05)
    parser.add_argument("--p95-kl-threshold", type=float, default=0.20)
    parser.add_argument("--top15-overlap-threshold", type=float, default=0.80)
    parser.add_argument("--top1-match-threshold", type=float, default=0.50)
    parser.add_argument("--out-json", default="/tmp/cosyvoice3_ane/quality_gate.json")
    return parser.parse_args()


def _nested_first_list(value, name: str) -> list[int]:
    if value is None:
        raise ValueError(f"missing {name}")
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [int(x) for x in value[0]]
    return [int(x) for x in value]


def _make_causal_mask(current_pos: int, context_length: int) -> np.ndarray:
    mask = np.full((1, 1, 1, context_length), -np.inf, dtype=np.float16)
    mask[:, :, :, : current_pos + 1] = 0.0
    return mask


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    m = np.max(x64)
    z = x64 - m
    return z - np.log(np.exp(z).sum())


def _kl_from_logits(ref: np.ndarray, cand: np.ndarray) -> float:
    ref_logp = _log_softmax(ref)
    cand_logp = _log_softmax(cand)
    p = np.exp(ref_logp)
    return float(np.sum(p * (ref_logp - cand_logp)))


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def _rank_of(scores: np.ndarray, token: int) -> int:
    # 1-based rank.  This avoids sorting the whole vocab for every step.
    return int(np.sum(scores > scores[token]) + 1)


def _load_coreml_model(path: Path, compute_units: str):
    import coremltools as ct

    cu_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    }
    try:
        return ct.models.MLModel(str(path), compute_units=cu_map[compute_units])
    except TypeError:
        # Older/newer coremltools stateful models can be picky; default loading is
        # still valid for the fidelity gate.
        return ct.models.MLModel(str(path))


def _mlx_reference_scores(backend: MlxQwenSpeechLM, text: torch.Tensor, prompt_speech_token: torch.Tensor,
                          generated: list[int], steps: int) -> list[np.ndarray]:
    mx = backend.mx
    text_ids = mx.array(text.detach().cpu().numpy().astype(np.int32, copy=False))
    text_emb = backend.model.model.embed_tokens(text_ids)
    sos_emb = backend.speech_embedding[backend.sos].reshape(1, 1, -1)
    task_id_emb = backend.speech_embedding[backend.task_id].reshape(1, 1, -1)
    if prompt_speech_token.shape[1] != 0:
        prompt_ids = mx.array(prompt_speech_token.detach().cpu().numpy().astype(np.int32, copy=False))
        prompt_speech_emb = backend.speech_embedding[prompt_ids]
        lm_input = mx.concatenate([sos_emb, text_emb, task_id_emb, prompt_speech_emb], axis=1)
    else:
        lm_input = mx.concatenate([sos_emb, text_emb, task_id_emb], axis=1)

    cache = [backend.KVCache() for _ in range(backend.num_layers)]
    scores_out: list[np.ndarray] = []
    for i in range(steps):
        hidden = backend.model.model(None, cache=cache, input_embeddings=lm_input)
        last = hidden[:, -1, :]
        scores = last @ backend.decoder_w.T
        if backend.decoder_b is not None:
            scores = scores + backend.decoder_b
        scores_out.append(np.array(scores[0].astype(mx.float32)))
        lm_input = backend.speech_embedding[int(generated[i])].reshape(1, 1, -1)
    return scores_out


def _iter_host_inputs(cosy_lm: torch.nn.Module, text: torch.Tensor, prompt_speech_token: torch.Tensor,
                      generated_prefix: Iterable[int]) -> tuple[np.ndarray, int]:
    text_emb = _embed_tokens(cosy_lm, text)
    sos_emb = cosy_lm.speech_embedding.weight[cosy_lm.sos].reshape(1, 1, -1)
    task_id_emb = cosy_lm.speech_embedding.weight[cosy_lm.task_id].reshape(1, 1, -1)
    if prompt_speech_token.shape[1] != 0:
        prompt_speech_emb = cosy_lm.speech_embedding(prompt_speech_token)
    else:
        prompt_speech_emb = torch.zeros(1, 0, cosy_lm.llm_input_size, dtype=text_emb.dtype, device=text_emb.device)
    prompt = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_emb], dim=1)
    generated_embs = [cosy_lm.speech_embedding.weight[int(tok)].reshape(1, 1, -1) for tok in generated_prefix]
    full = torch.concat([prompt, *generated_embs], dim=1) if generated_embs else prompt
    return full.detach().cpu().numpy().astype(np.float16), int(prompt.shape[1])


def _ane_scores(mlmodel, cosy_lm: torch.nn.Module, text: torch.Tensor, prompt_speech_token: torch.Tensor,
                generated: list[int], steps: int, context_length: int) -> list[np.ndarray]:
    # To produce N teacher-forced score vectors, run the prompt plus the first
    # N-1 generated tokens.  Scores after the prompt predict generated[0].
    host_inputs, prompt_len = _iter_host_inputs(cosy_lm, text, prompt_speech_token, generated[: max(0, steps - 1)])
    if host_inputs.shape[1] > context_length:
        raise SystemExit(f"context too short: need {host_inputs.shape[1]} positions, have {context_length}")

    state = mlmodel.make_state()
    scores_out: list[np.ndarray] = []
    decoder = cosy_lm.llm_decoder.cpu().eval()
    with torch.inference_mode():
        for pos in range(host_inputs.shape[1]):
            inputs = {
                "hidden_states": host_inputs[:, pos : pos + 1, :],
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": _make_causal_mask(pos, context_length),
                "current_pos": np.array([pos], dtype=np.int32),
            }
            output = mlmodel.predict(inputs, state)
            hidden = output["output_hidden_states"]
            if pos >= prompt_len - 1:
                hidden_t = torch.from_numpy(hidden.astype(np.float32, copy=False))
                scores = decoder(hidden_t).squeeze(0).squeeze(0).detach().cpu().float().numpy()
                scores_out.append(scores)
                if len(scores_out) >= steps:
                    break
    return scores_out


def main() -> int:
    args = parse_args()
    token_path = Path(args.tokens_json).expanduser().resolve()
    mlpackage = Path(args.mlpackage).expanduser().resolve()
    if not token_path.exists():
        raise SystemExit(f"missing token dump: {token_path}")
    if not mlpackage.exists():
        raise SystemExit(f"missing CoreML part-2 mlpackage: {mlpackage}")

    dump = json.loads(token_path.read_text(encoding="utf-8"))
    if args.chunk_index < 0 or args.chunk_index >= len(dump):
        raise SystemExit(f"chunk-index {args.chunk_index} out of range for {len(dump)} chunks")
    chunk = dump[args.chunk_index]
    text_tokens = _nested_first_list(chunk["prompt_text_tokens"], "prompt_text_tokens") + _nested_first_list(chunk["text_tokens"], "text_tokens")
    prompt_tokens = _nested_first_list(chunk["llm_prompt_speech_token"], "llm_prompt_speech_token")
    generated = _nested_first_list(chunk["generated_speech_token"], "generated_speech_token")
    steps = len(generated) if args.max_steps <= 0 else min(args.max_steps, len(generated))
    if steps <= 0:
        raise SystemExit("no generated tokens to gate")
    prompt_len = 1 + len(text_tokens) + 1 + len(prompt_tokens)
    needed_context = prompt_len + max(0, steps - 1)
    if needed_context > args.context_length:
        raise SystemExit(f"context too short for requested gate: need {needed_context}, have {args.context_length}")

    print(
        "QUALITY_GATE_THRESHOLDS "
        f"mean_kl<={args.mean_kl_threshold} p95_kl<={args.p95_kl_threshold} "
        f"top15_overlap>={args.top15_overlap_threshold} top1_match>={args.top1_match_threshold}"
    )
    print(
        "QUALITY_GATE_INPUT "
        f"chunk={args.chunk_index} steps={steps}/{len(generated)} prompt_len={prompt_len} context={args.context_length} "
        f"mlpackage={mlpackage}"
    )

    cosyvoice = AutoModel(model_dir=str(Path(args.model_dir).expanduser().resolve()), load_trt=False, load_vllm=False, fp16=False)
    apply_lora_adapter(cosyvoice, Path(args.adapter_dir).expanduser().resolve(), merge_lora=True)
    cosy_lm = cosyvoice.model.llm.cpu().eval()
    cosy_lm._mlx_dtype = args.mlx_dtype

    text = torch.tensor([text_tokens], dtype=torch.long)
    prompt_speech_token = torch.tensor([prompt_tokens], dtype=torch.long)

    t0 = time.time()
    mlx_backend = MlxQwenSpeechLM(cosy_lm)
    ref_scores = _mlx_reference_scores(mlx_backend, text, prompt_speech_token, generated, steps)
    print(f"MLX_REFERENCE_DONE steps={len(ref_scores)} wall={time.time() - t0:.2f}s")

    t1 = time.time()
    mlmodel = _load_coreml_model(mlpackage, args.compute_units)
    ane_scores = _ane_scores(mlmodel, cosy_lm, text, prompt_speech_token, generated, steps, args.context_length)
    print(f"ANE_PART2_DONE steps={len(ane_scores)} wall={time.time() - t1:.2f}s compute_units={args.compute_units}")

    if len(ref_scores) != len(ane_scores):
        raise SystemExit(f"score count mismatch: ref={len(ref_scores)} ane={len(ane_scores)}")

    kls = []
    top1_matches = []
    top5_overlaps = []
    top15_overlaps = []
    top50_overlaps = []
    teacher_ref_ranks = []
    teacher_ane_ranks = []
    for i, (ref, cand) in enumerate(zip(ref_scores, ane_scores)):
        kls.append(_kl_from_logits(ref, cand))
        ref_top1 = int(np.argmax(ref))
        cand_top1 = int(np.argmax(cand))
        top1_matches.append(1.0 if ref_top1 == cand_top1 else 0.0)
        for k, bucket in ((5, top5_overlaps), (15, top15_overlaps), (50, top50_overlaps)):
            a = set(int(x) for x in _top_indices(ref, k))
            b = set(int(x) for x in _top_indices(cand, k))
            bucket.append(len(a & b) / float(k))
        teacher = int(generated[i])
        teacher_ref_ranks.append(_rank_of(ref, teacher))
        teacher_ane_ranks.append(_rank_of(cand, teacher))

    kls_arr = np.array(kls, dtype=np.float64)
    summary = {
        "status": "pass",
        "chunk_index": args.chunk_index,
        "steps": steps,
        "prompt_len": prompt_len,
        "context_length": args.context_length,
        "mean_kl": float(kls_arr.mean()),
        "p50_kl": float(np.percentile(kls_arr, 50)),
        "p95_kl": float(np.percentile(kls_arr, 95)),
        "max_kl": float(kls_arr.max()),
        "top1_match_rate": float(np.mean(top1_matches)),
        "top5_overlap_mean": float(np.mean(top5_overlaps)),
        "top15_overlap_mean": float(np.mean(top15_overlaps)),
        "top50_overlap_mean": float(np.mean(top50_overlaps)),
        "teacher_ref_rank_median": float(np.median(teacher_ref_ranks)),
        "teacher_ane_rank_median": float(np.median(teacher_ane_ranks)),
        "teacher_ref_rank_p95": float(np.percentile(teacher_ref_ranks, 95)),
        "teacher_ane_rank_p95": float(np.percentile(teacher_ane_ranks, 95)),
        "thresholds": {
            "mean_kl": args.mean_kl_threshold,
            "p95_kl": args.p95_kl_threshold,
            "top15_overlap": args.top15_overlap_threshold,
            "top1_match": args.top1_match_threshold,
        },
        "mlpackage": str(mlpackage),
        "tokens_json": str(token_path),
    }
    failures = []
    if summary["mean_kl"] > args.mean_kl_threshold:
        failures.append("mean_kl")
    if summary["p95_kl"] > args.p95_kl_threshold:
        failures.append("p95_kl")
    if summary["top15_overlap_mean"] < args.top15_overlap_threshold:
        failures.append("top15_overlap")
    if summary["top1_match_rate"] < args.top1_match_threshold:
        failures.append("top1_match")
    if failures:
        summary["status"] = "fail"
        summary["failures"] = failures

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("QUALITY_GATE_SUMMARY " + json.dumps(summary, sort_keys=True))
    print(f"QUALITY_GATE_{summary['status'].upper()} out_json={out_json}")
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
