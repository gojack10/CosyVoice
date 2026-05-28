#!/usr/bin/env python3
"""Single-process CosyVoice3 LoRA smoke train.

Purpose: verify the local manifest -> parquet -> model forward/backward path without
CUDA/DDP. This is a smoke test, not the recommended production trainer. It uses
MPS if available, otherwise CPU, freezes the base LLM, applies PEFT LoRA, and
saves adapter-style files after a small number of steps.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.dataset.dataset import Dataset  # noqa: E402
from cosyvoice.utils.common import IGNORE_ID, th_accuracy  # noqa: E402


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _embed_tokens(cosy_llm: torch.nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    """Return Qwen token embeddings for both base and PEFT-wrapped Qwen."""
    qwen = cosy_llm.llm.model
    if hasattr(qwen, "base_model"):
        return qwen.base_model.model.model.embed_tokens(token_ids)
    return qwen.model.embed_tokens(token_ids)


def _lora_compatible_forward(self, batch: dict, device: torch.device):
    text_token = batch["text_token"].to(device)
    text_token_len = batch["text_token_len"].to(device)
    text_token_emb = _embed_tokens(self, text_token)

    if "speech_token" not in batch:
        raise RuntimeError("smoke_lora_single requires precomputed speech_token in parquet")
    speech_token = batch["speech_token"].to(device)
    speech_token_len = batch["speech_token_len"].to(device)
    speech_token_emb = self.speech_embedding(speech_token)

    sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
    task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

    instruct_token = batch["instruct_token"].to(device)
    instruct_token_len = batch["instruct_token_len"].to(device)
    instruct_token_emb = _embed_tokens(self, instruct_token)
    lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
        sos_emb,
        text_token,
        text_token_emb,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_emb,
        speech_token_len,
        instruct_token,
        instruct_token_emb,
        instruct_token_len,
    )
    lm_target = lm_target.to(device)
    lm_output, _ = self.llm(lm_input, lm_input_len.to(device))
    logits = self.llm_decoder(lm_output)
    loss = self.criterion_ce(logits, lm_target)
    acc = th_accuracy(logits.view(-1, self.llm_decoder.out_features), lm_target, ignore_label=IGNORE_ID)
    return {"loss": loss, "acc": acc}


def apply_lora(model: torch.nn.Module, r: int, alpha: int, dropout: float, targets: list[str]) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad = False
    model.llm.model = get_peft_model(
        model.llm.model,
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    model.forward = types.MethodType(_lora_compatible_forward, model)
    for p in model.speech_embedding.parameters():
        p.requires_grad = True
    for p in model.llm_decoder.parameters():
        p.requires_grad = True
    return model


def save_adapter(model: torch.nn.Module, out_dir: Path, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")
    peft_state = model_cpu.llm.model.state_dict()
    lora_state = {k: v for k, v in peft_state.items() if "lora_" in k}
    if not lora_state:
        raise RuntimeError("no lora_ tensors found after smoke training")
    torch.save(lora_state, out_dir / "lora_weights.pt")
    torch.save(model_cpu.speech_embedding.state_dict(), out_dir / "speech_embedding.pt")
    torch.save(model_cpu.llm_decoder.state_dict(), out_dir / "llm_decoder.pt")
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"format": "cosyvoice3_lora_adapter", "config": metadata}, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    parser.add_argument("--config", default=str(ROOT_DIR / "local_voice_pipeline/conf/cosyvoice3_smoke.yaml"))
    parser.add_argument("--train-data", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/train.data.list"))
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "local_voice_pipeline/outputs/lora_smoke_adapter"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,down_proj")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if (model_dir / "llm.rl.pt").exists():
        print("note: llm.rl.pt exists but this script loads base llm.pt", file=sys.stderr)
    checkpoint = model_dir / "llm.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing base checkpoint: {checkpoint}")
    train_data = Path(args.train_data).expanduser().resolve()
    if not train_data.exists():
        raise SystemExit(f"missing train data list: {train_data}")

    # Do not set onnx_path here: the local smoke uses precomputed speech_token
    # and embeddings in parquet. Setting onnx_path would make CosyVoice3LM try
    # to initialize the upstream CUDA-only online speech-token extractor.
    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    with open(args.config, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN"),
                "flow": None,
                "hift": None,
                "hifigan": None,
            },
        )

    dataset = Dataset(str(train_data), data_pipeline=configs["data_pipeline"], mode="train", gan=False, dpo=False, shuffle=False, partition=False)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    device = choose_device(args.device)
    print(f"using device={device}")
    model = configs["llm"]
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict({k: v for k, v in state.items() if isinstance(v, torch.Tensor)}, strict=False)
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, targets)
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable_params={sum(p.numel() for p in trainable_params)} total_params={sum(p.numel() for p in model.parameters())}")
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)

    losses = []
    for step, batch in enumerate(loader, 1):
        if step > args.max_steps:
            break
        optim.zero_grad(set_to_none=True)
        loss_dict = model(batch, device)
        loss = loss_dict["loss"]
        loss.backward()
        optim.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step} loss={losses[-1]:.6f} acc={float(loss_dict['acc']):.6f}")

    if not losses:
        raise SystemExit("no batches produced by train data")
    metadata = {
        "base_model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "base_checkpoint": str(checkpoint),
        "steps": len(losses),
        "losses": losses,
        "device": str(device),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": targets,
        "smoke_only": True,
    }
    save_adapter(model, Path(args.out_dir).expanduser().resolve(), metadata)
    print(f"wrote smoke adapter to {Path(args.out_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
