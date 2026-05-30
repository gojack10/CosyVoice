#!/usr/bin/env python3
"""Synthesize speech with a CosyVoice3 LoRA adapter checkpoint.

Loads the base CosyVoice3 model, applies a local LoRA adapter checkpoint saved by
train_lora_single_mps.py, then runs zero-shot inference with a prompt wav.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import types
from pathlib import Path

import numpy as np

import torch
import torchaudio
from peft import LoraConfig, get_peft_model

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

DEFAULT_TEXT = """SiftText is what I use to solve complex sprawling problems with AI. It helps me to 10x my speed in any problem solving domain, and I aim to use it to solve any problem BTO throws me during my internship. It's a platform purpose built for AI agents, so that they work with the problem in an epistemic manner that vanilla notetaking tools such as Notion or Obsidian do not achieve. I plan on using it to keep track of my projects and document invariants. Constraints, ruled out paths, resolved items, etc. Of course, PHI and proprietary company information will not be stored here, its strictly for tracking the project as it evolves."""
DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_temperature_sampler(temperature: float, top_p: float, top_k: int, win_size: int, tau_r: float,
                             force_cpu: bool = False, top_p_margin: float = 0.0):
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    def sample_from_scores(scores: torch.Tensor) -> int:
        if temperature == 0:
            return int(scores.argmax().item())
        scaled = scores / temperature
        probs = scaled.softmax(dim=0)
        k = min(top_k, probs.numel())
        top_probs, top_indices = torch.topk(probs, k=k, sorted=True)
        # Match the original nucleus loop: include the first token that crosses
        # top_p, but never include more than top_k candidates.
        cum_probs = torch.cumsum(top_probs, dim=0)
        keep_count = int((cum_probs < (top_p - top_p_margin)).sum().item()) + 1
        keep_count = max(1, min(keep_count, k))
        cand_probs = top_probs[:keep_count]
        cand_probs = cand_probs / cand_probs.sum()
        sample_pos = cand_probs.multinomial(1, replacement=True)
        return int(top_indices[sample_pos].item())

    def sampler(weighted_scores: torch.Tensor, decoded_tokens: list[int], sampling: int):
        if force_cpu and weighted_scores.device.type != "cpu":
            weighted_scores = weighted_scores.detach().cpu()
        top_id = sample_from_scores(weighted_scores)
        if decoded_tokens:
            # decoded_tokens is already a Python list of ints.  Avoid creating a
            # tiny MPS tensor and synchronizing every generated token just to
            # count repetitions; this is exactly equivalent to the tensor path.
            rep_num = decoded_tokens[-win_size:].count(top_id)
            if rep_num >= win_size * tau_r:
                scores = weighted_scores.clone()
                scores[top_id] = -float("inf")
                if temperature == 0:
                    top_id = int(scores.argmax().item())
                else:
                    top_id = int((scores / temperature).softmax(dim=0).multinomial(1, replacement=True).item())
        return top_id

    return sampler


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def patch_qwen_sdpa_no_kv_contiguous() -> None:
    """Avoid HF's unconditional key/value contiguous copies for MPS cache views."""
    from transformers.integrations.sdpa_attention import repeat_kv
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def sdpa_attention_forward_no_kv_contig(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs,
    ):
        if hasattr(module, "num_key_value_groups"):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        causal_mask = attention_mask
        if attention_mask is not None and causal_mask.ndim == 4:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]
        # Query is tiny in autoregressive decode, but keep it contiguous for the
        # SDPA frontend. Do not force K/V contiguous: MpsSliceCache returns
        # prefix views into preallocated cache storage, and copying that history
        # every layer/token defeats the cache.
        query = query.contiguous()
        if is_causal is None:
            is_causal = query.shape[2] > 1 and causal_mask is None
        if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
            is_causal = is_causal.item()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        return attn_output.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward_no_kv_contig
    print("patched Qwen SDPA attention to avoid K/V contiguous cache copies")


def _embed_tokens(cosy_llm: torch.nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    qwen = cosy_llm.llm.model

    def descend(obj, *names):
        for name in names:
            obj = getattr(obj, name, None)
            if obj is None:
                return None
        return obj

    candidates = [
        qwen,
        descend(qwen, "model"),
        descend(qwen, "base_model"),
        descend(qwen, "base_model", "model"),
        descend(qwen, "base_model", "model", "model"),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "embed_tokens"):
            return candidate.embed_tokens(token_ids)
    raise AttributeError(f"could not locate Qwen embed_tokens on {type(qwen).__name__}")


@torch.inference_mode()
def _lora_compatible_inference(
    self,
    text: torch.Tensor,
    text_len: torch.Tensor,
    prompt_text: torch.Tensor,
    prompt_text_len: torch.Tensor,
    prompt_speech_token: torch.Tensor,
    prompt_speech_token_len: torch.Tensor,
    embedding: torch.Tensor,
    sampling: int = 25,
    max_token_text_ratio: float = 20,
    min_token_text_ratio: float = 2,
    uuid: str = "",
):
    device = text.device
    text = torch.concat([prompt_text, text], dim=1)
    text_len = text_len + prompt_text_len
    if self.__class__.__name__ == "CosyVoice3LM":
        assert 151646 in text, "<|endofprompt|> not detected in CosyVoice3 text or prompt_text"
    forced_tokens = getattr(self, "_forced_speech_tokens", None)
    if forced_tokens is not None:
        for token in forced_tokens:
            yield int(token)
        return
    mlx_backend = getattr(self, "_mlx_backend", None)
    ane_backend = getattr(self, "_ane_backend", None)
    if mlx_backend is not None and ane_backend is not None:
        raise RuntimeError("choose only one accelerated LLM backend: MLX or ANE")
    llm_backend = mlx_backend or ane_backend
    if llm_backend is not None:
        min_token_text_ratio = float(getattr(self, "_min_token_text_ratio", min_token_text_ratio))
        max_token_text_ratio = float(getattr(self, "_max_token_text_ratio", max_token_text_ratio))
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        prefix_tokens = getattr(self, "_forced_speech_token_prefix", None)
        live_prefix_len = int(getattr(self, "_live_speech_token_prefix_len", 0) or 0)
        if prefix_tokens is None and live_prefix_len > 0:
            text_emb = _embed_tokens(self, text)
            sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
            task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
            if prompt_speech_token_len != 0:
                prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
            else:
                prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text_emb.dtype).to(device)
            lm_input = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)
            prefix_cap = min(live_prefix_len, max_len)
            prefix_tokens = [int(t) for t in self.inference_wrapper(lm_input, sampling, min_len, prefix_cap, uuid)]
            if bool(getattr(self, "_reset_seed_after_live_prefix", True)):
                set_seed(int(getattr(self, "_sampling_seed", 1986)))
            if len(prefix_tokens) < prefix_cap:
                for token in prefix_tokens:
                    yield token
                return
        for token in llm_backend.generate(text, prompt_speech_token, min_len, max_len, prefix_tokens=prefix_tokens):
            yield token
        return
    text_emb = _embed_tokens(self, text)
    if self.__class__.__name__ == "CosyVoice3LM":
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
    else:
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
    if prompt_speech_token_len != 0:
        prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
    else:
        prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text_emb.dtype).to(device)
    lm_input = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)
    min_token_text_ratio = float(getattr(self, "_min_token_text_ratio", min_token_text_ratio))
    max_token_text_ratio = float(getattr(self, "_max_token_text_ratio", max_token_text_ratio))
    min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
    max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
    for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
        yield token


def read_state(adapter_dir: Path) -> dict:
    for name in ("train_state.json", "config.json"):
        path = adapter_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if name == "config.json" and "config" in data:
            return data["config"]
        return data
    return {}


def apply_lora_adapter(cosyvoice, adapter_dir: Path, merge_lora: bool = True) -> None:
    state = read_state(adapter_dir)
    targets = state.get("lora_target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"]
    lora_config = LoraConfig(
        r=int(state.get("lora_r", 32)),
        lora_alpha=int(state.get("lora_alpha", 64)),
        lora_dropout=float(state.get("lora_dropout", 0.1)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(targets),
    )
    llm = cosyvoice.model.llm
    llm.llm.model = get_peft_model(llm.llm.model, lora_config)
    missing, unexpected = llm.llm.model.load_state_dict(torch.load(adapter_dir / "lora_weights.pt", map_location="cpu"), strict=False)
    llm.speech_embedding.load_state_dict(torch.load(adapter_dir / "speech_embedding.pt", map_location="cpu"), strict=True)
    llm.llm_decoder.load_state_dict(torch.load(adapter_dir / "llm_decoder.pt", map_location="cpu"), strict=True)
    if merge_lora and hasattr(llm.llm.model, "merge_and_unload"):
        llm.llm.model = llm.llm.model.merge_and_unload()
        print("merged LoRA adapter into base LLM for inference")
    llm.inference = types.MethodType(_lora_compatible_inference, llm)
    print(f"loaded adapter={adapter_dir}")
    print(f"lora missing={len(missing)} unexpected={len(unexpected)}")


def remove_hift_weight_norm(cosyvoice) -> None:
    hift = cosyvoice.model.hift
    if hasattr(hift, "remove_weight_norm"):
        try:
            hift.remove_weight_norm()
            print("removed HiFT weight norm parametrizations for inference")
        except ValueError as exc:
            # Already removed is harmless when experimenting in a long-lived process.
            print(f"skipped HiFT weight norm removal: {exc}")


def move_model(cosyvoice, device: torch.device, skip_llm_device: bool = False,
               flow_device: torch.device | None = None, hift_device: torch.device | None = None,
               flow_conditioner_device: torch.device | None = None,
               flow_decoder_device: torch.device | None = None) -> None:
    flow_device = flow_device or device
    hift_device = hift_device or device
    flow_conditioner_device = flow_conditioner_device or flow_device
    flow_decoder_device = flow_decoder_device or flow_device
    cosyvoice.model.device = flow_device
    if skip_llm_device:
        # MLX owns the Qwen/speech-token weights; keeping the duplicate PyTorch
        # LLM off MPS avoids extra MPS heap pressure while preserving inference.
        cosyvoice.model.llm.eval()
    else:
        cosyvoice.model.llm.to(device).eval()
    cosyvoice.model.flow.to(flow_device).eval()
    if all(hasattr(cosyvoice.model.flow, name) for name in ("input_embedding", "spk_embed_affine_layer", "pre_lookahead_layer", "decoder")):
        cosyvoice.model.flow.input_embedding.to(flow_conditioner_device).eval()
        cosyvoice.model.flow.spk_embed_affine_layer.to(flow_conditioner_device).eval()
        cosyvoice.model.flow.pre_lookahead_layer.to(flow_conditioner_device).eval()
        cosyvoice.model.flow.decoder.to(flow_decoder_device).eval()
    cosyvoice.model.hift.to(hift_device).eval()
    print(
        f"moved inference model to llm_device={device} flow_device={flow_device} "
        f"flow_conditioner_device={flow_conditioner_device} flow_decoder_device={flow_decoder_device} "
        f"hift_device={hift_device} skip_torch_llm_device={skip_llm_device}"
    )


def split_text(text: str) -> list[str]:
    text = " ".join(text.strip().split())
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    # Keep short fragments attached to neighbors where sensible.
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < 35:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged or [text]


# Unison doubler / chorus presets (center_ms, depth_ms, rate_hz, phase, gain).
# medium is the user-approved VEGA "two voices at once" amount.
DOUBLER_PRESETS = {
    "off": [],
    "subtle": [(20.0, 1.5, 0.6, 0.0, 0.40)],
    "medium": [(18.0, 3.0, 0.8, 0.0, 0.50), (26.0, 2.5, 1.1, float(np.pi), 0.45)],
    "strong": [(16.0, 4.0, 0.9, 0.0, 0.65), (28.0, 3.5, 1.3, float(np.pi), 0.60)],
}


def apply_doubler(speech: torch.Tensor, sr: int, amount: str) -> torch.Tensor:
    """Final post-stage: add modulated-delay, detuned copies of the dry voice and
    remix (the VEGA 'two voices' effect). amount='off' is a no-op passthrough."""
    voices = DOUBLER_PRESETS.get(amount, [])
    if not voices:
        return speech
    x = speech.squeeze(0).detach().cpu().numpy().astype(np.float64)
    n = x.shape[0]
    t = np.arange(n)
    out = x.copy()
    for center, depth, rate, phase, gain in voices:
        delay = (center + depth * np.sin(2 * np.pi * rate * t / sr + phase)) * sr / 1000.0
        out = out + gain * np.interp(t - delay, t, x, left=0.0, right=0.0)
    out = out / (np.max(np.abs(out)) + 1e-12) * 0.97
    return torch.from_numpy(out.astype(np.float32)).unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    parser.add_argument("--adapter-dir", default=str(ROOT_DIR / "local_voice_pipeline/exp/cosyvoice3/llm_lora/mps_single/best"))
    parser.add_argument("--flow-ckpt", default=str(ROOT_DIR / "local_voice_pipeline/exp/cosyvoice3/flow_ft/run1/final/flow.pt"), help="finetuned flow .pt loaded into cosyvoice.model.flow (default: the VEGA flow finetune that fixes timbre); pass '' for base flow")
    parser.add_argument("--hift-ckpt", default="", help="finetuned hift .pt to load into cosyvoice.model.hift; empty keeps the base hift")
    parser.add_argument("--double", default="medium", choices=("off", "subtle", "medium", "strong"), help="unison doubler post-stage = the VEGA two-voices effect (default: medium)")
    parser.add_argument("--prompt-wav", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/audio_24k/vega_0002.wav"))
    parser.add_argument("--prompt-transcript", default="After running diagnostics on the Praetor suit, it appears that I can activate optional challenges that, when completed, will assist in upgrading your arsenal at an accelerated pace.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--out", default=str(ROOT_DIR / "local_voice_pipeline/outputs/sifttext_lora_best.wav"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--flow-device", default=None, help="override device for flow/token2mel backend")
    parser.add_argument("--flow-conditioner-device", default=None, help="override device for flow token conditioning/pre-lookahead")
    parser.add_argument("--flow-decoder-device", default=None, help="override device for flow DiT/CFM decoder")
    parser.add_argument("--hift-device", default=None, help="override device for HiFT/vocoder backend")
    parser.add_argument("--sampling", type=int, default=25, help="legacy CosyVoice sampling arg; top-k is controlled by --top-k")
    # Default to the best-sounding ablation: t070_p075_k15_cool_conservative.
    parser.add_argument("--temperature", type=float, default=0.70)
    parser.add_argument("--top-p", type=float, default=0.75)
    parser.add_argument("--top-p-margin", type=float, default=0.0, help="subtract a tiny margin from top-p cutoff to reduce backend numeric boundary flips")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--ras-win-size", type=int, default=10)
    parser.add_argument("--ras-tau-r", type=float, default=0.1)
    parser.add_argument("--cpu-sampling", action="store_true", help="copy logits to CPU and use CPU torch.multinomial for exact CPU-style sampling")
    parser.add_argument("--seed", type=int, default=1986, help="negative disables explicit seeding")
    parser.add_argument("--no-merge-lora", action="store_true", help="keep PEFT LoRA modules unmerged for debugging")
    parser.add_argument("--mlx-llm", action="store_true", help="run CosyVoice3 Qwen speech-token decode in MLX")
    parser.add_argument("--mlx-dtype", default="float32", choices=("float32", "float16", "bfloat16"), help="dtype for MLX Qwen weights/activations")
    parser.add_argument("--mlx-native-sampling", action="store_true", help="sample top-k/top-p directly in MLX instead of CPU torch-style sampling")
    parser.add_argument("--ane-llm", action="store_true", help="run CosyVoice3 Qwen speech-token decode through Anemll/CoreML part-2 on ANE")
    parser.add_argument("--ane-mlpackage", default="/tmp/cosyvoice3_ane/coreml/cosyvoice3_qwen2_FFN_chunk_01of01.mlpackage", help="CoreML mlpackage produced by ane_convert_part2.py")
    parser.add_argument("--ane-context-length", type=int, default=1024, help="fixed CoreML KV context length used at conversion")
    parser.add_argument("--ane-compute-units", default="cpu_and_ne", choices=("all", "cpu_and_ne", "cpu_only", "cpu_and_gpu"), help="CoreML compute units for ANE backend")
    parser.add_argument("--flow-steps", type=int, default=10, help="flow diffusion Euler steps; lower is faster but quality-sensitive")
    parser.add_argument("--flow-cfg-rate", type=float, default=None, help="override flow classifier-free guidance rate")
    parser.add_argument("--profile-phases", action="store_true", help="synchronize and print LLM/flow/HiFT phase timings")
    parser.add_argument("--keep-hift-weight-norm", action="store_true", help="do not remove HiFT weight norm parametrizations")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true", help="use CosyVoice streaming token2wav path to overlap MLX LLM and audio backend")
    parser.add_argument("--silence-ms", type=int, default=250)
    parser.add_argument("--min-token-text-ratio", type=float, default=6.0)
    parser.add_argument("--max-token-text-ratio", type=float, default=25.0)
    parser.add_argument("--mask-all-stop-before-min-len", action="store_true", help="during MLX min_len, mask every stop token instead of only base EOS")
    parser.add_argument("--prefix-tts", action="store_true", help="also prefix each target chunk with the CosyVoice3 instruction")
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--chunk-separator", default=None, help="split --text on this literal separator instead of automatic sentence splitting")
    parser.add_argument("--dump-tokens-json", default=None, help="write per-chunk frontend and generated speech tokens for debugging")
    parser.add_argument("--force-tokens-json", default=None, help="bypass LLM and synthesize from per-chunk generated_speech_token values")
    parser.add_argument("--force-token-prefix-json", default=None, help="prime MLX LLM with first N generated_speech_token values from this dump")
    parser.add_argument("--force-token-prefix-len", type=int, default=0, help="number of prefix tokens per chunk to force before MLX continues")
    parser.add_argument("--live-token-prefix-len", type=int, default=0, help="generate this many initial speech tokens with the PyTorch LLM at runtime, then let MLX continue")
    parser.add_argument("--no-reset-seed-after-live-prefix", action="store_true", help="do not reset RNG before MLX continuation after live prefix generation")
    args = parser.parse_args()
    if args.mlx_llm and args.ane_llm:
        raise SystemExit("choose only one accelerated LLM backend: --mlx-llm or --ane-llm")

    model_dir = Path(args.model_dir).expanduser().resolve()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    prompt_wav = Path(args.prompt_wav).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    for required in (model_dir / "llm.pt", adapter_dir / "lora_weights.pt", adapter_dir / "speech_embedding.pt", adapter_dir / "llm_decoder.pt", prompt_wav):
        if not required.exists():
            raise SystemExit(f"missing required path: {required}")

    set_seed(args.seed)
    device = choose_device(args.device)
    flow_device = choose_device(args.flow_device) if args.flow_device else device
    flow_conditioner_device = choose_device(args.flow_conditioner_device) if args.flow_conditioner_device else flow_device
    flow_decoder_device = choose_device(args.flow_decoder_device) if args.flow_decoder_device else flow_device
    hift_device = choose_device(args.hift_device) if args.hift_device else device
    if any(d.type == "mps" for d in (device, flow_device, flow_conditioner_device, flow_decoder_device, hift_device)):
        patch_qwen_sdpa_no_kv_contiguous()
    cosyvoice = AutoModel(model_dir=str(model_dir), load_trt=False, load_vllm=False, fp16=False)
    apply_lora_adapter(cosyvoice, adapter_dir, merge_lora=not args.no_merge_lora)
    if args.flow_ckpt:
        flow_ckpt = Path(args.flow_ckpt).expanduser().resolve()
        if not flow_ckpt.exists():
            raise SystemExit(f"missing flow ckpt: {flow_ckpt}")
        flow_state = {k: v for k, v in torch.load(flow_ckpt, map_location="cpu").items() if isinstance(v, torch.Tensor)}
        f_missing, f_unexpected = cosyvoice.model.flow.load_state_dict(flow_state, strict=False)
        print(f"loaded finetuned flow {flow_ckpt}: missing={len(f_missing)} unexpected={len(f_unexpected)}")
    if args.hift_ckpt:
        hift_ckpt = Path(args.hift_ckpt).expanduser().resolve()
        if not hift_ckpt.exists():
            raise SystemExit(f"missing hift ckpt: {hift_ckpt}")
        hift_state = {k: v for k, v in torch.load(hift_ckpt, map_location="cpu").items() if isinstance(v, torch.Tensor)}
        h_missing, h_unexpected = cosyvoice.model.hift.load_state_dict(hift_state, strict=False)
        print(f"loaded finetuned hift {hift_ckpt}: missing={len(h_missing)} unexpected={len(h_unexpected)}")
    cosyvoice.model.llm._min_token_text_ratio = args.min_token_text_ratio
    cosyvoice.model.llm._max_token_text_ratio = args.max_token_text_ratio
    cosyvoice.model.llm._mask_all_stop_before_min_len = args.mask_all_stop_before_min_len
    cosyvoice.model.llm._sampling_temperature = args.temperature
    cosyvoice.model.llm._sampling_top_p = args.top_p
    cosyvoice.model.llm._sampling_top_p_margin = args.top_p_margin
    cosyvoice.model.llm._sampling_top_k = args.top_k
    cosyvoice.model.llm._sampling_ras_win_size = args.ras_win_size
    cosyvoice.model.llm._sampling_ras_tau_r = args.ras_tau_r
    cosyvoice.model.llm._sampling_seed = args.seed
    cosyvoice.model.llm._mlx_dtype = args.mlx_dtype
    cosyvoice.model.llm._ane_mlpackage = args.ane_mlpackage
    cosyvoice.model.llm._ane_context_length = args.ane_context_length
    cosyvoice.model.llm._ane_compute_units = args.ane_compute_units
    cosyvoice.model.llm._live_speech_token_prefix_len = args.live_token_prefix_len
    cosyvoice.model.llm._reset_seed_after_live_prefix = not args.no_reset_seed_after_live_prefix
    cosyvoice.model.llm._mlx_native_sampling = args.mlx_native_sampling
    cosyvoice.model.llm.sampling = make_temperature_sampler(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        win_size=args.ras_win_size,
        tau_r=args.ras_tau_r,
        force_cpu=args.cpu_sampling,
        top_p_margin=args.top_p_margin,
    )
    print(
        f"sampling temperature={args.temperature} top_p={args.top_p} top_p_margin={args.top_p_margin} top_k={args.top_k} "
        f"ras_win_size={args.ras_win_size} ras_tau_r={args.ras_tau_r} seed={args.seed}"
    )
    if args.mlx_llm:
        from local_voice_pipeline.mlx_qwen_backend import attach_mlx_qwen
        attach_mlx_qwen(cosyvoice.model.llm)
    if args.ane_llm:
        from local_voice_pipeline.ane_qwen_backend import attach_ane_qwen
        attach_ane_qwen(cosyvoice.model.llm)
    cosyvoice.model._profile_inference = args.profile_phases
    cosyvoice.model.flow._inference_timesteps = args.flow_steps
    if args.flow_cfg_rate is not None:
        cosyvoice.model.flow.decoder._inference_cfg_rate = args.flow_cfg_rate
    if not args.keep_hift_weight_norm:
        remove_hift_weight_norm(cosyvoice)
    # AutoModel/config loading sets a fixed seed internally; reset here so
    # --seed actually controls speech-token sampling in this harness.
    set_seed(args.seed)
    move_model(
        cosyvoice,
        device,
        skip_llm_device=((args.mlx_llm or args.ane_llm) and args.live_token_prefix_len <= 0),
        flow_device=flow_device,
        hift_device=hift_device,
        flow_conditioner_device=flow_conditioner_device,
        flow_decoder_device=flow_decoder_device,
    )

    token_dump = []
    if args.dump_tokens_json:
        orig_token2wav = cosyvoice.model.token2wav

        def token2wav_recording(*t2w_args, **t2w_kwargs):
            token_tensor = t2w_kwargs.get('token')
            if token_tensor is None and t2w_args:
                token_tensor = t2w_args[0]
            cosyvoice.model._last_debug_tts_token = None if token_tensor is None else token_tensor.detach().cpu().to(torch.int32).tolist()
            return orig_token2wav(*t2w_args, **t2w_kwargs)

        cosyvoice.model.token2wav = token2wav_recording

    prompt_text = DEFAULT_INSTRUCT + args.prompt_transcript
    if args.chunk_separator is not None:
        chunks = [part.strip() for part in args.text.split(args.chunk_separator) if part.strip()]
    else:
        chunks = [args.text] if args.no_split else split_text(args.text)
    forced_token_dump = None
    if args.force_tokens_json:
        forced_token_dump = json.loads(Path(args.force_tokens_json).expanduser().read_text(encoding="utf-8"))
        if len(forced_token_dump) != len(chunks):
            raise SystemExit(f"force token chunk count mismatch: {len(forced_token_dump)} tokens vs {len(chunks)} chunks")
    forced_prefix_dump = None
    if args.force_token_prefix_json:
        forced_prefix_dump = json.loads(Path(args.force_token_prefix_json).expanduser().read_text(encoding="utf-8"))
        if len(forced_prefix_dump) != len(chunks):
            raise SystemExit(f"force prefix chunk count mismatch: {len(forced_prefix_dump)} tokens vs {len(chunks)} chunks")
    print(f"synthesizing {len(chunks)} chunk(s)")
    pieces: list[torch.Tensor] = []
    silence_len = int(cosyvoice.sample_rate * args.silence_ms / 1000)
    silence = torch.zeros(1, silence_len)
    started = time.time()
    for idx, chunk in enumerate(chunks, 1):
        tts_text = (DEFAULT_INSTRUCT + chunk) if args.prefix_tts else chunk
        print(f"chunk {idx}/{len(chunks)} chars={len(chunk)}: {chunk}", flush=True)
        chunk_start = time.time()
        # Fast exact path for this harness: text_frontend=False and chunks are
        # already selected, so bypass inference_zero_shot's tqdm/log wrapper and
        # call the same frontend/model methods directly inside the timed region.
        prompt_text_norm = cosyvoice.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
        model_input = cosyvoice.frontend.frontend_zero_shot(
            tts_text,
            prompt_text_norm,
            str(prompt_wav),
            cosyvoice.sample_rate,
            '',
        )
        if forced_token_dump is not None:
            cosyvoice.model.llm._forced_speech_tokens = forced_token_dump[idx - 1]["generated_speech_token"][0]
        else:
            cosyvoice.model.llm._forced_speech_tokens = None
        if forced_prefix_dump is not None and args.force_token_prefix_len > 0:
            cosyvoice.model.llm._forced_speech_token_prefix = forced_prefix_dump[idx - 1]["generated_speech_token"][0][:args.force_token_prefix_len]
        else:
            cosyvoice.model.llm._forced_speech_token_prefix = None
        outputs = list(cosyvoice.model.tts(**model_input, stream=args.stream, speed=args.speed))
        if args.dump_tokens_json:
            token_dump.append({
                "chunk_index": idx,
                "chunk_text": chunk,
                "tts_text": tts_text,
                "text_tokens": model_input["text"].detach().cpu().to(torch.int32).tolist(),
                "prompt_text_tokens": model_input["prompt_text"].detach().cpu().to(torch.int32).tolist(),
                "llm_prompt_speech_token": model_input["llm_prompt_speech_token"].detach().cpu().to(torch.int32).tolist(),
                "flow_prompt_speech_token": model_input["flow_prompt_speech_token"].detach().cpu().to(torch.int32).tolist(),
                "generated_speech_token": getattr(cosyvoice.model, "_last_debug_tts_token", None),
            })
        if not outputs:
            raise SystemExit(f"no output for chunk {idx}")
        speech = torch.cat([item["tts_speech"] for item in outputs], dim=1).cpu()
        print(f"chunk {idx} duration={speech.shape[1] / cosyvoice.sample_rate:.2f}s wall={time.time() - chunk_start:.2f}s", flush=True)
        if pieces:
            pieces.append(silence)
        pieces.append(speech)

    speech = torch.cat(pieces, dim=1)
    speech = apply_doubler(speech, cosyvoice.sample_rate, args.double)
    print(f"applied doubler={args.double}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), speech, cosyvoice.sample_rate)
    metadata = {
        "out": str(out),
        "sample_rate": cosyvoice.sample_rate,
        "duration_sec": speech.shape[1] / cosyvoice.sample_rate,
        "adapter_dir": str(adapter_dir),
        "flow_ckpt": args.flow_ckpt,
        "hift_ckpt": args.hift_ckpt,
        "double": args.double,
        "prompt_wav": str(prompt_wav),
        "prompt_text": prompt_text,
        "text": args.text,
        "chunks": chunks,
        "device": str(device),
        "flow_device": str(flow_device),
        "flow_conditioner_device": str(flow_conditioner_device),
        "flow_decoder_device": str(flow_decoder_device),
        "hift_device": str(hift_device),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_p_margin": args.top_p_margin,
        "top_k": args.top_k,
        "ras_win_size": args.ras_win_size,
        "ras_tau_r": args.ras_tau_r,
        "seed": args.seed,
        "merge_lora": not args.no_merge_lora,
        "mlx_llm": args.mlx_llm,
        "mlx_dtype": args.mlx_dtype,
        "mlx_native_sampling": args.mlx_native_sampling,
        "ane_llm": args.ane_llm,
        "ane_mlpackage": args.ane_mlpackage,
        "ane_context_length": args.ane_context_length,
        "ane_compute_units": args.ane_compute_units,
        "live_token_prefix_len": args.live_token_prefix_len,
        "reset_seed_after_live_prefix": not args.no_reset_seed_after_live_prefix,
        "flow_steps": args.flow_steps,
        "flow_cfg_rate": args.flow_cfg_rate,
        "mask_all_stop_before_min_len": args.mask_all_stop_before_min_len,
        "force_tokens_json": args.force_tokens_json,
        "force_token_prefix_json": args.force_token_prefix_json,
        "force_token_prefix_len": args.force_token_prefix_len,
        "chunk_separator": args.chunk_separator,
        "no_split": args.no_split,
        "profile_phases": args.profile_phases,
        "remove_hift_weight_norm": not args.keep_hift_weight_norm,
        "speed": args.speed,
        "stream": args.stream,
        "wall_sec": time.time() - started,
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dump_tokens_json:
        token_dump_path = Path(args.dump_tokens_json).expanduser().resolve()
        token_dump_path.parent.mkdir(parents=True, exist_ok=True)
        token_dump_path.write_text(json.dumps(token_dump, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote token dump {token_dump_path}")
    print(f"wrote {out} sample_rate={cosyvoice.sample_rate} duration={metadata['duration_sec']:.2f}s wall={metadata['wall_sec']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
