"""Experimental MLX backend for CosyVoice3 Qwen speech-token decoding."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


class MlxQwenSpeechLM:
    """Batch=1 inference-only MLX implementation of the CosyVoice3 LLM hot path."""

    def __init__(self, cosy_lm: torch.nn.Module):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from mlx_lm.models.qwen2 import Model, ModelArgs

        self.mx = mx
        self.KVCache = KVCache
        dtype_name = str(getattr(cosy_lm, "_mlx_dtype", "float32")).lower()
        dtype_map = {
            "float32": mx.float32,
            "fp32": mx.float32,
            "float16": mx.float16,
            "fp16": mx.float16,
            "bfloat16": mx.bfloat16,
            "bf16": mx.bfloat16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported MLX dtype {dtype_name!r}; use float32, float16, or bfloat16")
        self.dtype_name = dtype_name
        self.dtype = dtype_map[dtype_name]
        qwen = cosy_lm.llm.model
        cfg = qwen.config
        args = ModelArgs(
            model_type="qwen2",
            hidden_size=int(cfg.hidden_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            intermediate_size=int(cfg.intermediate_size),
            num_attention_heads=int(cfg.num_attention_heads),
            rms_norm_eps=float(cfg.rms_norm_eps),
            vocab_size=int(cfg.vocab_size),
            num_key_value_heads=int(cfg.num_key_value_heads),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rope_theta=float(cfg.rope_theta),
            rope_traditional=False,
            rope_scaling=cfg.rope_scaling,
            tie_word_embeddings=bool(cfg.tie_word_embeddings),
        )
        self.model = Model(args)
        weights = [
            (name, mx.array(tensor.detach().cpu().numpy(), dtype=self.dtype))
            for name, tensor in qwen.state_dict().items()
            if name.startswith("model.")
        ]
        self.model.load_weights(weights, strict=False)

        self.speech_embedding = mx.array(cosy_lm.speech_embedding.weight.detach().cpu().numpy(), dtype=self.dtype)
        self.decoder_w = mx.array(cosy_lm.llm_decoder.weight.detach().cpu().numpy(), dtype=self.dtype)
        self.decoder_b = None if cosy_lm.llm_decoder.bias is None else mx.array(cosy_lm.llm_decoder.bias.detach().cpu().numpy(), dtype=self.dtype)
        self.speech_token_size = int(cosy_lm.speech_token_size)
        self.sos = int(cosy_lm.sos)
        self.task_id = int(cosy_lm.task_id)
        self.stop_token_ids = set(int(x) for x in cosy_lm.stop_token_ids)
        self.mask_all_stop_before_min_len = bool(getattr(cosy_lm, "_mask_all_stop_before_min_len", False))
        self.native_sampling = bool(getattr(cosy_lm, "_mlx_native_sampling", False))
        self.seed = int(getattr(cosy_lm, "_sampling_seed", 1986))
        if self.native_sampling and self.seed >= 0:
            mx.random.seed(self.seed)
        self.num_layers = int(cfg.num_hidden_layers)
        self.temperature = float(getattr(cosy_lm, "_sampling_temperature", 0.70))
        self.top_p = float(getattr(cosy_lm, "_sampling_top_p", 0.75))
        self.top_p_margin = float(getattr(cosy_lm, "_sampling_top_p_margin", 0.0))
        self.top_k = int(getattr(cosy_lm, "_sampling_top_k", 15))
        self.ras_win_size = int(getattr(cosy_lm, "_sampling_ras_win_size", 10))
        self.ras_tau_r = float(getattr(cosy_lm, "_sampling_ras_tau_r", 0.1))
        to_eval = [self.speech_embedding, self.decoder_w]
        if self.decoder_b is not None:
            to_eval.append(self.decoder_b)
        mx.eval(*to_eval)
        print(
            "initialized MLX Qwen speech LM "
            f"layers={self.num_layers} hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
            f"kv_heads={cfg.num_key_value_heads} dtype={self.dtype_name} native_sampling={self.native_sampling} "
            f"params={len(weights)} sos={self.sos} task_id={self.task_id}"
        )

    def _sample_from_scores(self, scores: np.ndarray) -> int:
        # Match the PyTorch CPU sampler used by the last good generation.  The
        # previous MLX sampler used numpy float64 + np.random.choice, which
        # changes the stochastic token path even when logits are close.
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
                # Match CosyVoice sampling_ids exactly: during min_len only the base
                # eos id is masked, not all CosyVoice3 reserved stop ids.
                scores[self.speech_token_size] = -np.inf
        top_id = self._sample_from_scores(scores)
        if decoded_tokens:
            rep_num = decoded_tokens[-self.ras_win_size:].count(top_id)
            if rep_num >= self.ras_win_size * self.ras_tau_r:
                scores = scores.copy()
                scores[top_id] = -np.inf
                # Match make_temperature_sampler's RAS fallback exactly: after
                # suppressing the repeated token, draw from the full softmax.
                top_id = self._sample_full_softmax(scores)
        return top_id

    def _mask_scores_mx(self, scores):
        mx = self.mx
        ids = mx.arange(scores.shape[0])
        if self.mask_all_stop_before_min_len:
            mask = mx.zeros(scores.shape, dtype=mx.bool_)
            for stop_id in self.stop_token_ids:
                if 0 <= stop_id < scores.shape[0]:
                    mask = mx.logical_or(mask, ids == stop_id)
        else:
            mask = ids == self.speech_token_size
        return mx.where(mask, mx.array(-float("inf"), dtype=scores.dtype), scores)

    def _sample_from_scores_mx(self, scores):
        mx = self.mx
        if self.temperature == 0:
            return int(np.array(mx.argmax(scores)).item())
        k = min(self.top_k, scores.shape[0])
        top_indices = mx.argpartition(-scores, kth=k - 1)[:k]
        top_scores = mx.take(scores, top_indices)
        order = mx.argsort(-top_scores)
        top_indices = mx.take(top_indices, order)
        top_scores = mx.take(top_scores, order)
        top_probs = mx.softmax(top_scores / self.temperature, axis=0)
        cum_probs = mx.cumsum(top_probs, axis=0)
        keep_count = mx.sum(cum_probs < (self.top_p - self.top_p_margin)) + 1
        keep_mask = mx.arange(k) < keep_count
        logits = mx.where(keep_mask, top_scores / self.temperature, mx.array(-float("inf"), dtype=top_scores.dtype))
        sample_pos = mx.random.categorical(logits, axis=0)
        return int(np.array(mx.take(top_indices, sample_pos)).item())

    def _sample_full_softmax_mx(self, scores):
        mx = self.mx
        if self.temperature == 0:
            return int(np.array(mx.argmax(scores)).item())
        sample_pos = mx.random.categorical(scores / self.temperature, axis=0)
        return int(np.array(sample_pos).item())

    def _sample_native(self, scores, decoded_tokens: list[int], ignore_eos: bool) -> int:
        if ignore_eos:
            scores = self._mask_scores_mx(scores)
        top_id = self._sample_from_scores_mx(scores)
        if decoded_tokens:
            rep_num = decoded_tokens[-self.ras_win_size:].count(top_id)
            if rep_num >= self.ras_win_size * self.ras_tau_r:
                mx = self.mx
                ids = mx.arange(scores.shape[0])
                scores = mx.where(ids == top_id, mx.array(-float("inf"), dtype=scores.dtype), scores)
                top_id = self._sample_full_softmax_mx(scores)
        return top_id

    def generate(self, text: torch.Tensor, prompt_speech_token: torch.Tensor, min_len: int, max_len: int,
                 prefix_tokens: list[int] | None = None) -> Iterable[int]:
        mx = self.mx
        text_ids = mx.array(text.detach().cpu().numpy().astype(np.int32, copy=False))
        text_emb = self.model.model.embed_tokens(text_ids)
        sos_emb = self.speech_embedding[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token.shape[1] != 0:
            prompt_ids = mx.array(prompt_speech_token.detach().cpu().numpy().astype(np.int32, copy=False))
            prompt_speech_emb = self.speech_embedding[prompt_ids]
            lm_input = mx.concatenate([sos_emb, text_emb, task_id_emb, prompt_speech_emb], axis=1)
        else:
            lm_input = mx.concatenate([sos_emb, text_emb, task_id_emb], axis=1)

        cache = [self.KVCache() for _ in range(self.num_layers)]
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
            prefix_ids = mx.array(np.array(prefix_tokens, dtype=np.int32).reshape(1, -1))
            prefix_emb = self.speech_embedding[prefix_ids]
            lm_input = mx.concatenate([lm_input, prefix_emb], axis=1)
        while len(out_tokens) < max_len:
            i = len(out_tokens)
            hidden = self.model.model(None, cache=cache, input_embeddings=lm_input)
            last = hidden[:, -1, :]
            scores = last @ self.decoder_w.T
            if self.decoder_b is not None:
                scores = scores + self.decoder_b
            if self.native_sampling:
                top_id = self._sample_native(scores[0], out_tokens, ignore_eos=i < min_len)
            else:
                scores_np = np.array(scores[0].astype(mx.float32))
                top_id = self._sample(scores_np, out_tokens, ignore_eos=i < min_len)
            if top_id in self.stop_token_ids:
                break
            yield top_id
            out_tokens.append(top_id)
            lm_input = self.speech_embedding[top_id].reshape(1, 1, -1)


def attach_mlx_qwen(cosy_lm: torch.nn.Module) -> None:
    cosy_lm._mlx_backend = MlxQwenSpeechLM(cosy_lm)
