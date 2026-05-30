"""CoreML/ANE backend for CosyVoice3 Qwen speech-token decoding.

This is intentionally transformer-only: host code assembles Qwen text embeddings +
CosyVoice speech embeddings, calls the Anemll part-2 CoreML model, then applies
CosyVoice's llm_decoder and CPU-style sampler on the host.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _embed_tokens(cosy_lm: torch.nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    qwen = cosy_lm.llm.model

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
            device = candidate.embed_tokens.weight.device
            return candidate.embed_tokens(token_ids.to(device))
    raise AttributeError(f"could not locate Qwen embed_tokens on {type(qwen).__name__}")


class CoreMLQwenSpeechLM:
    """Batch=1 inference-only CoreML implementation of the CosyVoice3 LLM hot path."""

    def __init__(self, cosy_lm: torch.nn.Module):
        import coremltools as ct

        self.cosy_lm = cosy_lm.cpu().eval()
        mlpackage = Path(getattr(cosy_lm, "_ane_mlpackage", "/tmp/cosyvoice3_ane/coreml/cosyvoice3_qwen2_FFN_chunk_01of01.mlpackage")).expanduser().resolve()
        if not mlpackage.exists():
            raise FileNotFoundError(f"missing ANE/CoreML Qwen part-2 mlpackage: {mlpackage}")
        self.mlpackage = mlpackage
        self.context_length = int(getattr(cosy_lm, "_ane_context_length", 1024))
        compute_units = str(getattr(cosy_lm, "_ane_compute_units", "cpu_and_ne")).lower()
        cu_map = {
            "all": ct.ComputeUnit.ALL,
            "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        }
        if compute_units not in cu_map:
            raise ValueError(f"unsupported CoreML compute units {compute_units!r}")
        self.compute_units = compute_units
        self.model = ct.models.MLModel(str(mlpackage), compute_units=cu_map[compute_units])
        # Verify state creation at startup so failures happen before rendering.
        self.model.make_state()

        self.speech_embedding = self.cosy_lm.speech_embedding.cpu().eval()
        self.decoder = self.cosy_lm.llm_decoder.cpu().eval()
        self.speech_token_size = int(self.cosy_lm.speech_token_size)
        self.sos = int(self.cosy_lm.sos)
        self.task_id = int(self.cosy_lm.task_id)
        self.stop_token_ids = set(int(x) for x in self.cosy_lm.stop_token_ids)
        self.mask_all_stop_before_min_len = bool(getattr(self.cosy_lm, "_mask_all_stop_before_min_len", False))
        self.temperature = float(getattr(self.cosy_lm, "_sampling_temperature", 0.70))
        self.top_p = float(getattr(self.cosy_lm, "_sampling_top_p", 0.75))
        self.top_p_margin = float(getattr(self.cosy_lm, "_sampling_top_p_margin", 0.0))
        self.top_k = int(getattr(self.cosy_lm, "_sampling_top_k", 15))
        self.ras_win_size = int(getattr(self.cosy_lm, "_sampling_ras_win_size", 10))
        self.ras_tau_r = float(getattr(self.cosy_lm, "_sampling_ras_tau_r", 0.1))
        print(
            "initialized CoreML/ANE Qwen speech LM "
            f"mlpackage={self.mlpackage} context={self.context_length} compute_units={self.compute_units} "
            f"sos={self.sos} task_id={self.task_id}"
        )

    def _make_causal_mask(self, current_pos: int) -> np.ndarray:
        mask = np.full((1, 1, 1, self.context_length), -np.inf, dtype=np.float16)
        mask[:, :, :, : current_pos + 1] = 0.0
        return mask

    def _step(self, hidden: np.ndarray, current_pos: int, state) -> np.ndarray:
        if current_pos >= self.context_length:
            raise RuntimeError(f"ANE Qwen context exceeded: {current_pos + 1} > {self.context_length}")
        output = self.model.predict(
            {
                "hidden_states": hidden.astype(np.float16, copy=False),
                "position_ids": np.array([current_pos], dtype=np.int32),
                "causal_mask": self._make_causal_mask(current_pos),
                "current_pos": np.array([current_pos], dtype=np.int32),
            },
            state,
        )
        return output["output_hidden_states"]

    def _decode_scores(self, hidden: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            hidden_t = torch.from_numpy(hidden.astype(np.float32, copy=False))
            scores = self.decoder(hidden_t).squeeze(0).squeeze(0).detach().cpu().float().numpy()
        return scores

    def _sample_from_scores(self, scores: np.ndarray) -> int:
        # Match the PyTorch CPU sampler used by the MLX backend and last-good runs.
        scores_t = torch.from_numpy(scores.astype(np.float32, copy=False))
        if self.temperature == 0:
            return int(scores_t.argmax().item())
        probs = (scores_t / self.temperature).softmax(dim=0)
        k = min(self.top_k, probs.numel())
        top_probs, top_indices = torch.topk(probs, k=k, sorted=True)
        cum_probs = torch.cumsum(top_probs, dim=0)
        keep_count = int((cum_probs < (self.top_p - self.top_p_margin)).sum().item()) + 1
        keep_count = max(1, min(keep_count, k))
        cand_probs = top_probs[:keep_count]
        cand_probs = cand_probs / cand_probs.sum()
        sample_pos = cand_probs.multinomial(1, replacement=True)
        return int(top_indices[sample_pos].item())

    def _sample_full_softmax(self, scores: np.ndarray) -> int:
        scores_t = torch.from_numpy(scores.astype(np.float32, copy=False))
        if self.temperature == 0:
            return int(scores_t.argmax().item())
        return int((scores_t / self.temperature).softmax(dim=0).multinomial(1, replacement=True).item())

    def _sample(self, scores: np.ndarray, decoded_tokens: list[int], ignore_eos: bool) -> int:
        if ignore_eos:
            scores = scores.copy()
            if self.mask_all_stop_before_min_len:
                for stop_id in self.stop_token_ids:
                    if 0 <= stop_id < scores.shape[0]:
                        scores[stop_id] = -np.inf
            else:
                scores[self.speech_token_size] = -np.inf
        top_id = self._sample_from_scores(scores)
        if decoded_tokens:
            rep_num = decoded_tokens[-self.ras_win_size:].count(top_id)
            if rep_num >= self.ras_win_size * self.ras_tau_r:
                scores = scores.copy()
                scores[top_id] = -np.inf
                top_id = self._sample_full_softmax(scores)
        return top_id

    def _prompt_embeddings(self, text: torch.Tensor, prompt_speech_token: torch.Tensor,
                           prefix_tokens: list[int] | None) -> np.ndarray:
        with torch.inference_mode():
            text_emb = _embed_tokens(self.cosy_lm, text.detach().cpu())
            sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
            task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
            if prompt_speech_token.shape[1] != 0:
                prompt_speech_emb = self.speech_embedding(prompt_speech_token.detach().cpu())
                pieces = [sos_emb, text_emb.cpu(), task_id_emb, prompt_speech_emb]
            else:
                pieces = [sos_emb, text_emb.cpu(), task_id_emb]
            if prefix_tokens:
                prefix_ids = torch.tensor([prefix_tokens], dtype=torch.long)
                pieces.append(self.speech_embedding(prefix_ids))
            lm_input = torch.concat(pieces, dim=1)
        return lm_input.detach().cpu().numpy().astype(np.float16)

    def generate(self, text: torch.Tensor, prompt_speech_token: torch.Tensor, min_len: int, max_len: int,
                 prefix_tokens: list[int] | None = None) -> Iterable[int]:
        state = self.model.make_state()
        out_tokens: list[int] = []
        if prefix_tokens:
            prefix_tokens = [int(t) for t in prefix_tokens[:max_len]]
            for token in prefix_tokens:
                if token in self.stop_token_ids:
                    return
                yield token
                out_tokens.append(token)
            if len(out_tokens) >= max_len:
                return
        lm_input = self._prompt_embeddings(text, prompt_speech_token, prefix_tokens)
        hidden = None
        for pos in range(lm_input.shape[1]):
            hidden = self._step(lm_input[:, pos : pos + 1, :], pos, state)
        current_pos = lm_input.shape[1]
        while len(out_tokens) < max_len:
            if hidden is None:
                raise RuntimeError("ANE Qwen produced no hidden state")
            i = len(out_tokens)
            scores = self._decode_scores(hidden)
            top_id = self._sample(scores, out_tokens, ignore_eos=i < min_len)
            if top_id in self.stop_token_ids:
                break
            yield top_id
            out_tokens.append(top_id)
            with torch.inference_mode():
                next_hidden = self.speech_embedding.weight[top_id].reshape(1, 1, -1).detach().cpu().numpy().astype(np.float16)
            hidden = self._step(next_hidden, current_pos, state)
            current_pos += 1


def attach_ane_qwen(cosy_lm: torch.nn.Module) -> None:
    cosy_lm._ane_backend = CoreMLQwenSpeechLM(cosy_lm)
