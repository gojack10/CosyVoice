#!/usr/bin/env python3
"""CosyVoice3 LLM LoRA training entrypoint.

This mirrors cosyvoice/bin/train.py but applies PEFT LoRA to the nested Qwen2
LLM after loading the base checkpoint and before optimizer construction.

It is intended for CUDA/Linux training. For first adaptation, use this only if
you explicitly want adapter training instead of full LLM SFT. It still starts
from base llm.pt, not llm.rl.pt.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

# Keep deepspeed support aligned with upstream train.py. The official
# requirements install deepspeed on Linux; macOS prep/inference envs do not.
import deepspeed  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from cosyvoice.utils.common import IGNORE_ID, th_accuracy  # noqa: E402
from cosyvoice.utils.executor import Executor  # noqa: E402
from cosyvoice.utils.losses import DPOLoss  # noqa: E402
from cosyvoice.utils.train_utils import (  # noqa: E402
    check_modify_and_save_config,
    init_dataset_and_dataloader,
    init_distributed,
    init_optimizer_and_scheduler,
    init_summarywriter,
    save_model,
    wrap_cuda_model,
)


def get_args():
    parser = argparse.ArgumentParser(description="LoRA training for CosyVoice3 LLM")
    parser.add_argument("--train_engine", default="torch_ddp", choices=["torch_ddp", "deepspeed"])
    parser.add_argument("--model", required=True, choices=["llm"], help="Only CosyVoice3 llm LoRA is supported here")
    parser.add_argument("--ref_model", required=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--cv_data", required=True)
    parser.add_argument("--qwen_pretrain_path", required=False)
    parser.add_argument("--onnx_path", required=False)
    parser.add_argument("--checkpoint", required=True, help="Base llm.pt checkpoint; do not pass llm.rl.pt for first adaptation")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tensorboard_dir", default="tensorboard")
    parser.add_argument("--ddp.dist_backend", dest="dist_backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--prefetch", default=100, type=int)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--dpo", action="store_true", default=False)
    parser.add_argument("--deepspeed.save_states", dest="save_states", default="model_only", choices=["model_only", "model+optimizer"])
    parser.add_argument("--timeout", default=60, type=int)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,down_proj")
    parser.add_argument("--train_speech_embedding", action="store_true", default=True)
    parser.add_argument("--no_train_speech_embedding", dest="train_speech_embedding", action="store_false")
    parser.add_argument("--train_llm_decoder", action="store_true", default=True)
    parser.add_argument("--no_train_llm_decoder", dest="train_llm_decoder", action="store_false")
    parser.add_argument("--adapter_metadata", default="", help="Optional note copied into saved train_conf metadata")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def _embed_tokens(cosy_llm: torch.nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    qwen = cosy_llm.llm.model
    if hasattr(qwen, "base_model"):
        return qwen.base_model.model.model.embed_tokens(token_ids)
    return qwen.model.embed_tokens(token_ids)


def _lora_compatible_forward(self, batch: dict, device: torch.device):
    text_token = batch['text_token'].to(device)
    text_token_len = batch['text_token_len'].to(device)
    text_token_emb = _embed_tokens(self, text_token)

    if 'speech_token' not in batch:
        speech_token, speech_token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
    else:
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
    speech_token_emb = self.speech_embedding(speech_token)

    sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
    task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

    instruct_token = batch['instruct_token'].to(device)
    instruct_token_len = batch['instruct_token_len'].to(device)
    instruct_token_emb = _embed_tokens(self, instruct_token)
    lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
        sos_emb, text_token, text_token_emb, text_token_len, task_id_emb,
        speech_token, speech_token_emb, speech_token_len, instruct_token, instruct_token_emb, instruct_token_len)
    lm_target = lm_target.to(device)
    lm_output, _ = self.llm(lm_input, lm_input_len.to(device))
    logits = self.llm_decoder(lm_output)
    loss = self.criterion_ce(logits, lm_target)
    acc = th_accuracy(logits.view(-1, self.llm_decoder.out_features), lm_target, ignore_label=IGNORE_ID)
    return {'loss': loss, 'acc': acc}


def apply_lora(model: torch.nn.Module, args) -> torch.nn.Module:
    if not hasattr(model, "llm") or not hasattr(model.llm, "model"):
        raise TypeError("expected CosyVoice3LM with nested llm.model Qwen2ForCausalLM")

    # Freeze base CosyVoice3 LLM first. PEFT will re-enable LoRA params.
    for param in model.parameters():
        param.requires_grad = False

    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model.llm.model = get_peft_model(model.llm.model, lora_config)
    model.forward = types.MethodType(_lora_compatible_forward, model)

    if args.train_speech_embedding:
        for param in model.speech_embedding.parameters():
            param.requires_grad = True
    if args.train_llm_decoder:
        for param in model.llm_decoder.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info("LoRA trainable parameters: %s / %s (%.4f%%)", trainable, total, 100 * trainable / max(total, 1))
    return model


def save_lora_metadata(args, model_dir: str) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    metadata = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": [x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        "train_speech_embedding": args.train_speech_embedding,
        "train_llm_decoder": args.train_llm_decoder,
        "base_checkpoint": args.checkpoint,
        "note": args.adapter_metadata,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    import json

    with open(Path(model_dir) / "lora_train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


@record
def main():
    args = get_args()
    os.environ["onnx_path"] = args.onnx_path or ""
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    if args.checkpoint.endswith("llm.rl.pt"):
        raise SystemExit("Refusing llm.rl.pt for first voice adaptation; pass base llm.pt")
    if args.dpo:
        raise SystemExit("LoRA DPO/RL is intentionally disabled for first voice adaptation")

    override_dict = {k: None for k in ["flow", "hift", "hifigan"]}
    if args.qwen_pretrain_path is not None:
        override_dict["qwen_pretrain_path"] = args.qwen_pretrain_path
    with open(args.config, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs["train_conf"].update(vars(args))
    configs["train_conf"]["lora"] = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_modules": [x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        "train_speech_embedding": args.train_speech_embedding,
        "train_llm_decoder": args.train_llm_decoder,
    }

    init_distributed(args)
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = init_dataset_and_dataloader(args, configs, gan=False, dpo=args.dpo)
    configs = check_modify_and_save_config(args, configs)
    writer = init_summarywriter(args)

    model = configs[args.model]
    start_step, start_epoch = 0, -1
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)
    if "step" in state_dict:
        start_step = state_dict["step"]
    if "epoch" in state_dict:
        start_epoch = state_dict["epoch"]

    model = apply_lora(model, args)
    save_lora_metadata(args, args.model_dir)

    model = wrap_cuda_model(args, model)
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan=False)
    scheduler.set_step(start_step)

    info_dict = deepcopy(configs["train_conf"])
    info_dict["step"] = start_step
    info_dict["epoch"] = start_epoch
    save_model(model, "init", info_dict)

    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        ref_state_dict = torch.load(args.ref_model, map_location="cpu")
        ref_model.load_state_dict(_strip_module_prefix(ref_state_dict), strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        ref_model = wrap_cuda_model(args, ref_model)
    else:
        ref_model, dpo_loss = None, None

    executor = Executor(gan=False, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print(f"start step {start_step} start epoch {start_epoch}")
    for epoch in range(start_epoch + 1, info_dict["max_epoch"]):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=ref_model)
        dist.destroy_process_group(group_join)


if __name__ == "__main__":
    main()
