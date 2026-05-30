#!/usr/bin/env python3
"""Single-process CosyVoice3 LoRA trainer for Apple Silicon/MPS.

The official CosyVoice trainer in this checkout is CUDA/DDP-oriented. This
entrypoint trains a PEFT LoRA adapter plus the CosyVoice3 speech embedding and
LLM decoder on one process using MPS when available.

Design goals for a small voice dataset:
- no quality-reducing quantization or low precision shortcuts;
- granular adapter + optimizer checkpoints so any pre-overfit step is usable;
- full observability through JSONL logs, checkpoint index, and TensorBoard.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from hyperpyyaml import load_hyperpyyaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional at runtime
    SummaryWriter = None  # type: ignore[assignment]

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.dataset.dataset import Dataset  # noqa: E402
from cosyvoice.utils.common import IGNORE_ID, th_accuracy  # noqa: E402


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _embed_tokens(cosy_llm: torch.nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    qwen = cosy_llm.llm.model
    if hasattr(qwen, "base_model"):
        return qwen.base_model.model.model.embed_tokens(token_ids)
    return qwen.model.embed_tokens(token_ids)


def _lora_compatible_forward(self, batch: dict, device: torch.device):
    text_token = batch["text_token"].to(device)
    text_token_len = batch["text_token_len"].to(device)
    text_token_emb = _embed_tokens(self, text_token)

    if "speech_token" not in batch:
        raise RuntimeError("train_lora_single_mps requires precomputed speech_token in parquet")
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
    for param in model.parameters():
        param.requires_grad = False
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
    for param in model.speech_embedding.parameters():
        param.requires_grad = True
    for param in model.llm_decoder.parameters():
        param.requires_grad = True
    return model


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def _mps_metrics() -> dict[str, float]:
    if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        return {}
    out: dict[str, float] = {}
    mps_mod = getattr(torch, "mps", None)
    if mps_mod is None:
        return out
    for name, key in [
        ("current_allocated_memory", "mps_current_allocated_gb"),
        ("driver_allocated_memory", "mps_driver_allocated_gb"),
        ("recommended_max_memory", "mps_recommended_max_gb"),
    ]:
        fn = getattr(mps_mod, name, None)
        if callable(fn):
            try:
                out[key] = float(fn()) / 1024**3
            except Exception:
                pass
    return out


def runtime_metrics() -> dict[str, float]:
    # ru_maxrss is bytes on macOS and KiB on Linux. This job runs on macOS, but
    # keep a Linux fallback for local tests.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024.0
    return {"rss_gb": rss / 1024**3, **_mps_metrics()}


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    checkpoint_index: Path,
) -> None:
    """Save adapter artifacts plus optimizer/train state for exact resumption."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    peft_state = model.llm.model.state_dict()
    lora_state = {name: tensor.detach().cpu() for name, tensor in peft_state.items() if "lora_" in name}
    if not lora_state:
        raise RuntimeError("no lora_ tensors found while saving checkpoint")
    torch.save(lora_state, checkpoint_dir / "lora_weights.pt")
    torch.save(_cpu_state_dict(model.speech_embedding), checkpoint_dir / "speech_embedding.pt")
    torch.save(_cpu_state_dict(model.llm_decoder), checkpoint_dir / "llm_decoder.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    write_json(checkpoint_dir / "train_state.json", metadata)
    write_json(checkpoint_dir / "config.json", {"format": "cosyvoice3_lora_adapter", "config": metadata})
    append_jsonl(checkpoint_index, {"event": "checkpoint", "path": str(checkpoint_dir), **metadata})


def copy_checkpoint(src_dir: Path, dst_dir: Path) -> None:
    """Replace dst_dir with a copy of src_dir. Avoid symlinks for portability."""
    tmp_dir = dst_dir.with_name(dst_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(src_dir, tmp_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    tmp_dir.replace(dst_dir)


def load_resume(model: torch.nn.Module, optimizer: torch.optim.Optimizer, resume_dir: Path, load_optimizer: bool) -> dict[str, Any]:
    state_path = resume_dir / "train_state.json"
    if not state_path.exists():
        raise SystemExit(f"resume checkpoint missing train_state.json: {resume_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    model.llm.model.load_state_dict(torch.load(resume_dir / "lora_weights.pt", map_location="cpu"), strict=False)
    model.speech_embedding.load_state_dict(torch.load(resume_dir / "speech_embedding.pt", map_location="cpu"), strict=True)
    model.llm_decoder.load_state_dict(torch.load(resume_dir / "llm_decoder.pt", map_location="cpu"), strict=True)
    if load_optimizer and (resume_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location="cpu"))
    return state


def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            loss_dict = model(batch, device)
            losses.append(float(loss_dict["loss"].detach().cpu()))
            accs.append(float(loss_dict["acc"]))
    model.train()
    return {"loss": mean(losses), "acc": mean(accs), "batches": len(losses)}


def tensorboard_log(writer: Any, prefix: str, metrics: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None and not isinstance(value, bool):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def maybe_evaluate_and_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cv_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    run_metadata: dict[str, Any],
    out_dir: Path,
    log_path: Path,
    checkpoint_index: Path,
    writer: Any,
    epoch: int,
    global_step: int,
    last_train_loss: float | None,
    last_train_acc: float | None,
    best_eval_loss: float,
    reason: str,
) -> float:
    eval_metrics = run_eval(model, cv_loader, device, args.max_eval_batches)
    metrics = {
        "event": "eval",
        "reason": reason,
        "epoch": epoch,
        "step": global_step,
        "train_loss": last_train_loss,
        "train_acc": last_train_acc,
        "eval_loss": eval_metrics["loss"],
        "eval_acc": eval_metrics["acc"],
        "eval_batches": eval_metrics["batches"],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **runtime_metrics(),
    }
    append_jsonl(log_path, metrics)
    tensorboard_log(writer, "eval", {k.replace("eval_", ""): v for k, v in metrics.items()}, global_step)
    print(json.dumps(metrics, sort_keys=True), flush=True)

    eval_loss = eval_metrics["loss"]
    is_new_best = eval_loss is not None and eval_loss < best_eval_loss
    candidate_best = float(eval_loss) if is_new_best else best_eval_loss
    ckpt_metadata = {
        **run_metadata,
        **metrics,
        "best_eval_loss": candidate_best if math.isfinite(candidate_best) else None,
        "finished": False,
    }
    step_dir = out_dir / f"step_{global_step:06d}"
    save_checkpoint(model, optimizer, step_dir, ckpt_metadata, checkpoint_index)
    copy_checkpoint(step_dir, out_dir / "latest")
    if is_new_best:
        best_eval_loss = candidate_best
        copy_checkpoint(step_dir, out_dir / "best")
        write_json(out_dir / "best_metrics.json", ckpt_metadata)
        append_jsonl(log_path, {"event": "new_best", "step": global_step, "eval_loss": best_eval_loss, "path": str(out_dir / "best")})
    return best_eval_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CosyVoice3 LoRA on one Apple Silicon/MPS process")
    parser.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    parser.add_argument("--config", default=str(ROOT_DIR / "local_voice_pipeline/conf/cosyvoice3_sft.yaml"))
    parser.add_argument("--train-data", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/train.data.list"))
    parser.add_argument("--cv-data", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/dev.data.list"))
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "local_voice_pipeline/exp/cosyvoice3/llm_lora/mps_single"))
    parser.add_argument("--tensorboard-dir", default="", help="default: <out-dir>/tensorboard")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means full dev set")
    parser.add_argument("--eval-every-steps", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=1, help="independent of eval; eval steps are always saved")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 disables; measured in evals without dev-loss improvement")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accum-grad", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,down_proj")
    parser.add_argument("--resume-dir", default="")
    parser.add_argument("--no-resume-optimizer", action="store_true")
    parser.add_argument("--note", default="MPS single-process CosyVoice3 LoRA run")
    args = parser.parse_args()

    if args.accum_grad < 1:
        raise SystemExit("--accum-grad must be >= 1")
    if args.eval_every_steps < 1:
        raise SystemExit("--eval-every-steps must be >= 1")
    if args.save_every_steps < 1:
        raise SystemExit("--save-every-steps must be >= 1")

    # MPS fallback keeps quality-oriented fp32 training while letting unsupported
    # ops run on CPU rather than failing midway through a long run.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model_dir = Path(args.model_dir).expanduser().resolve()
    checkpoint = model_dir / "llm.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing base checkpoint: {checkpoint}")
    if (model_dir / "llm.rl.pt").exists():
        print("note: llm.rl.pt exists but this trainer loads base llm.pt", file=sys.stderr)

    train_data = Path(args.train_data).expanduser().resolve()
    cv_data = Path(args.cv_data).expanduser().resolve()
    for path in (train_data, cv_data):
        if not path.exists():
            raise SystemExit(f"missing data list: {path}")

    targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    with open(args.config, "r", encoding="utf-8") as file:
        configs = load_hyperpyyaml(
            file,
            overrides={
                "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN"),
                "flow": None,
                "hift": None,
                "hifigan": None,
            },
        )

    train_dataset = Dataset(str(train_data), data_pipeline=configs["data_pipeline"], mode="train", gan=False, dpo=False, shuffle=True, partition=False)
    cv_dataset = Dataset(str(cv_data), data_pipeline=configs["data_pipeline"], mode="dev", gan=False, dpo=False, shuffle=False, partition=False)
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    cv_loader = DataLoader(cv_dataset, batch_size=None, num_workers=0)

    device = choose_device(args.device)
    print(f"using device={device}", flush=True)
    model = configs["llm"]
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict({key: value for key, value in state.items() if isinstance(value, torch.Tensor)}, strict=False)
    del state

    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, targets)
    model.to(device)
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for param in trainable_params)
    total_count = sum(param.numel() for param in model.parameters())
    print(f"trainable_params={trainable_count} total_params={total_count}", flush=True)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    checkpoint_index = out_dir / "checkpoint_index.jsonl"
    tb_dir = Path(args.tensorboard_dir).expanduser().resolve() if args.tensorboard_dir else out_dir / "tensorboard"
    writer = SummaryWriter(str(tb_dir)) if SummaryWriter is not None else None

    run_metadata = {
        "base_model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "base_checkpoint": str(checkpoint),
        "config": str(Path(args.config).expanduser().resolve()),
        "train_data": str(train_data),
        "cv_data": str(cv_data),
        "out_dir": str(out_dir),
        "tensorboard_dir": str(tb_dir),
        "device": str(device),
        "epochs": args.epochs,
        "lr": args.lr,
        "accum_grad": args.accum_grad,
        "eval_every_steps": args.eval_every_steps,
        "save_every_steps": args.save_every_steps,
        "max_eval_batches": args.max_eval_batches,
        "early_stop_patience": args.early_stop_patience,
        "min_delta": args.min_delta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": targets,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "note": args.note,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(out_dir / "run_metadata.json", run_metadata)
    append_jsonl(log_path, {"event": "run_start", **run_metadata, **runtime_metrics()})
    if writer is not None:
        writer.add_text("run/metadata", json.dumps(run_metadata, indent=2, sort_keys=True), 0)

    global_step = 0
    best_eval_loss = math.inf
    no_improve_evals = 0
    resume_state: dict[str, Any] = {}
    if args.resume_dir:
        resume_state = load_resume(model, optimizer, Path(args.resume_dir).expanduser().resolve(), not args.no_resume_optimizer)
        global_step = int(resume_state.get("step", 0) or 0)
        previous_best = resume_state.get("best_eval_loss") or resume_state.get("eval_loss")
        if previous_best is not None:
            best_eval_loss = float(previous_best)
        append_jsonl(log_path, {"event": "resumed", "resume_dir": args.resume_dir, "step": global_step, "state": resume_state})

    optimizer.zero_grad(set_to_none=True)
    should_stop = False

    try:
        for epoch in range(int(resume_state.get("epoch", -1)) + 1, args.epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            epoch_losses: list[float] = []
            epoch_accs: list[float] = []
            pending_accum = 0
            last_train_loss: float | None = None
            last_train_acc: float | None = None

            for batch_idx, batch in enumerate(train_loader, 1):
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    should_stop = True
                    break

                loss_dict = model(batch, device)
                raw_loss = float(loss_dict["loss"].detach().cpu())
                raw_acc = float(loss_dict["acc"])
                (loss_dict["loss"] / args.accum_grad).backward()
                pending_accum += 1
                epoch_losses.append(raw_loss)
                epoch_accs.append(raw_acc)
                last_train_loss = raw_loss
                last_train_acc = raw_acc

                batch_event = {
                    "event": "train_batch",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "pending_accum": pending_accum,
                    "step": global_step,
                    "loss": raw_loss,
                    "acc": raw_acc,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    **runtime_metrics(),
                }
                append_jsonl(log_path, batch_event)
                tensorboard_log(writer, "train_batch", {"loss": raw_loss, "acc": raw_acc}, max(global_step * args.accum_grad + batch_idx, 0))

                if pending_accum >= args.accum_grad:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending_accum = 0
                    global_step += 1
                    step_event = {
                        "event": "optimizer_step",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "step": global_step,
                        "train_loss": last_train_loss,
                        "train_acc": last_train_acc,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        **runtime_metrics(),
                    }
                    append_jsonl(log_path, step_event)
                    tensorboard_log(writer, "train", {"loss": last_train_loss, "acc": last_train_acc, **runtime_metrics()}, global_step)
                    print(f"epoch={epoch} step={global_step} loss={last_train_loss:.6f} acc={last_train_acc:.6f}", flush=True)

                    should_eval = global_step % args.eval_every_steps == 0
                    should_save_only = global_step % args.save_every_steps == 0
                    if should_eval:
                        old_best = best_eval_loss
                        best_eval_loss = maybe_evaluate_and_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            cv_loader=cv_loader,
                            device=device,
                            args=args,
                            run_metadata={**run_metadata, "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None},
                            out_dir=out_dir,
                            log_path=log_path,
                            checkpoint_index=checkpoint_index,
                            writer=writer,
                            epoch=epoch,
                            global_step=global_step,
                            last_train_loss=last_train_loss,
                            last_train_acc=last_train_acc,
                            best_eval_loss=best_eval_loss,
                            reason="step",
                        )
                        if best_eval_loss < old_best - args.min_delta:
                            no_improve_evals = 0
                        else:
                            no_improve_evals += 1
                        if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                            append_jsonl(log_path, {"event": "early_stop", "step": global_step, "no_improve_evals": no_improve_evals})
                            should_stop = True
                            break
                    elif should_save_only:
                        ckpt_metadata = {
                            **run_metadata,
                            "event": "checkpoint_no_eval",
                            "epoch": epoch,
                            "step": global_step,
                            "train_loss": last_train_loss,
                            "train_acc": last_train_acc,
                            "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            **runtime_metrics(),
                        }
                        step_dir = out_dir / f"step_{global_step:06d}"
                        save_checkpoint(model, optimizer, step_dir, ckpt_metadata, checkpoint_index)
                        copy_checkpoint(step_dir, out_dir / "latest")

            if pending_accum > 0 and not should_stop:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                append_jsonl(log_path, {"event": "optimizer_step_partial", "epoch": epoch, "step": global_step, "pending_accum": pending_accum})
                best_eval_loss = maybe_evaluate_and_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    cv_loader=cv_loader,
                    device=device,
                    args=args,
                    run_metadata={**run_metadata, "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None},
                    out_dir=out_dir,
                    log_path=log_path,
                    checkpoint_index=checkpoint_index,
                    writer=writer,
                    epoch=epoch,
                    global_step=global_step,
                    last_train_loss=last_train_loss,
                    last_train_acc=last_train_acc,
                    best_eval_loss=best_eval_loss,
                    reason="partial_accum_epoch_end",
                )

            epoch_summary = {
                "event": "epoch_end",
                "epoch": epoch,
                "step": global_step,
                "train_loss": mean(epoch_losses),
                "train_acc": mean(epoch_accs),
                "train_batches": len(epoch_losses),
                "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
                "seconds": time.time() - epoch_start,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **runtime_metrics(),
            }
            append_jsonl(log_path, epoch_summary)
            tensorboard_log(writer, "epoch", epoch_summary, global_step)
            print(json.dumps(epoch_summary, sort_keys=True), flush=True)

            if should_stop:
                break

    except KeyboardInterrupt:
        interrupted_metadata = {
            **run_metadata,
            "event": "interrupted",
            "step": global_step,
            "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        append_jsonl(log_path, interrupted_metadata)
        save_checkpoint(model, optimizer, out_dir / "interrupted", interrupted_metadata, checkpoint_index)
        if writer is not None:
            writer.close()
        raise

    done_metadata = {
        **run_metadata,
        "event": "run_end",
        "finished": True,
        "step": global_step,
        "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **runtime_metrics(),
    }
    save_checkpoint(model, optimizer, out_dir / "final", done_metadata, checkpoint_index)
    copy_checkpoint(out_dir / "final", out_dir / "latest")
    append_jsonl(log_path, done_metadata)
    write_json(out_dir / "final_metrics.json", done_metadata)
    if writer is not None:
        writer.add_text("run/final", json.dumps(done_metadata, indent=2, sort_keys=True), global_step)
        writer.close()
    print(f"training complete; wrote checkpoints under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
