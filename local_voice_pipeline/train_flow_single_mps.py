#!/usr/bin/env python3
"""Single-process CosyVoice3 FLOW finetuner for Apple Silicon/MPS.

Sibling of train_lora_single_mps.py. Where that trainer adapts the LLM (speech
token generation), this one adapts the flow-matching model
(CausalMaskedDiffWithDiT). The flow renders speech tokens -> mel conditioned on
the speaker embedding, so it is the carrier of *timbre* and of the processed
"AI/robot effect" of the reference voice. The stock flow was trained on clean
human speech and regularizes the target toward a clean human read; finetuning it
on the target recovers that character.

Design goals for a small, fixed voice dataset:
- finetune a focused subset by default (decoder + spk_embed_affine_layer) so we
  capture timbre/effect without disturbing token->hidden alignment / timing;
- full fp32, no quantization shortcuts;
- sparse full-flow checkpoints (best/latest/final, each ~1.3 GB) plus an optional
  periodic snapshot, instead of the per-step dumps the LLM trainer used;
- JSONL + TensorBoard observability; eval on the held-out dev split.

The eval loss is only a proxy. User audition of rendered held-out lines is the
gate, consistent with the rest of this pipeline.
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
from pathlib import Path
from typing import Any

import torch
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional at runtime
    SummaryWriter = None  # type: ignore[assignment]

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.dataset.dataset import Dataset  # noqa: E402


# CausalMaskedDiffWithDiT submodules. The decoder (CFM/DiT) is the mel generator
# and the primary carrier of timbre/effect; spk_embed_affine_layer projects the
# speaker embedding into the conditioning. input_embedding / pre_lookahead_layer
# handle token semantics and alignment, so they stay frozen unless asked.
TRAINABLE_GROUPS: dict[str, list[str]] = {
    "all": ["input_embedding", "spk_embed_affine_layer", "pre_lookahead_layer", "decoder"],
    "decoder": ["decoder"],
    "decoder_spk": ["decoder", "spk_embed_affine_layer"],
    "decoder_spk_prelook": ["decoder", "spk_embed_affine_layer", "pre_lookahead_layer"],
}


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_trainable(model: torch.nn.Module, group: str) -> tuple[int, int]:
    if group not in TRAINABLE_GROUPS:
        raise SystemExit(f"unknown --trainable group '{group}'; choose from {sorted(TRAINABLE_GROUPS)}")
    for param in model.parameters():
        param.requires_grad = False
    for name in TRAINABLE_GROUPS[group]:
        sub = getattr(model, name, None)
        if sub is None:
            raise SystemExit(f"flow has no submodule '{name}' (needed for --trainable {group})")
        for param in sub.parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


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
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":  # ru_maxrss is bytes on macOS, KiB on Linux
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


def save_flow_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    checkpoint_index: Path,
    save_optimizer: bool,
) -> None:
    """Save the *full* flow state_dict so inference can load it directly into
    cosyvoice.model.flow. Each file is ~1.3 GB, so callers save sparingly."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu_state_dict(model), checkpoint_dir / "flow.pt")
    if save_optimizer:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    write_json(checkpoint_dir / "train_state.json", metadata)
    write_json(checkpoint_dir / "config.json", {"format": "cosyvoice3_flow_finetune", "config": metadata})
    append_jsonl(checkpoint_index, {"event": "checkpoint", "path": str(checkpoint_dir), **metadata})


def copy_checkpoint(src_dir: Path, dst_dir: Path) -> None:
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
    model.load_state_dict(torch.load(resume_dir / "flow.pt", map_location="cpu"), strict=False)
    if load_optimizer and (resume_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location="cpu"))
    return state


def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            loss_dict = model(batch, device)
            losses.append(float(loss_dict["loss"].detach().cpu()))
    model.train()
    return {"loss": mean(losses), "batches": len(losses)}


def tensorboard_log(writer: Any, prefix: str, metrics: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None and not isinstance(value, bool):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def evaluate_and_checkpoint(
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
        "eval_loss": eval_metrics["loss"],
        "eval_batches": eval_metrics["batches"],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **runtime_metrics(),
    }
    append_jsonl(log_path, metrics)
    tensorboard_log(writer, "eval", {"loss": eval_metrics["loss"]}, global_step)
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
    # Overwrite "latest" each eval; promote to "best" on improvement. Optional
    # periodic snapshot keeps a step-tagged copy for auditioning the trajectory.
    save_flow_checkpoint(model, optimizer, out_dir / "latest", ckpt_metadata, checkpoint_index, args.save_optimizer)
    if args.keep_step_every > 0 and global_step % args.keep_step_every == 0:
        copy_checkpoint(out_dir / "latest", out_dir / f"step_{global_step:06d}")
    if is_new_best:
        best_eval_loss = candidate_best
        copy_checkpoint(out_dir / "latest", out_dir / "best")
        write_json(out_dir / "best_metrics.json", ckpt_metadata)
        append_jsonl(log_path, {"event": "new_best", "step": global_step, "eval_loss": best_eval_loss, "path": str(out_dir / "best")})
    return best_eval_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Finetune the CosyVoice3 flow on one Apple Silicon/MPS process")
    parser.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    parser.add_argument("--config", default=str(ROOT_DIR / "local_voice_pipeline/conf/cosyvoice3_sft.yaml"))
    parser.add_argument("--train-data", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/train.data.list"))
    parser.add_argument("--cv-data", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/dev.data.list"))
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "local_voice_pipeline/exp/cosyvoice3/flow_ft/mps_single"))
    parser.add_argument("--tensorboard-dir", default="", help="default: <out-dir>/tensorboard")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means full dev set")
    parser.add_argument("--eval-every-steps", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=20, help="eval steps always checkpoint; this also checkpoints latest without eval")
    parser.add_argument("--keep-step-every", type=int, default=0, help="0 disables; else snapshot a step-tagged full-flow copy every N steps")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 disables; measured in evals without dev-loss improvement")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accum-grad", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=5.0, help="<=0 disables grad clipping")
    parser.add_argument("--trainable", default="decoder_spk", help=f"which flow submodules to train: {sorted(TRAINABLE_GROUPS)}")
    parser.add_argument("--save-optimizer", action="store_true", help="also save optimizer.pt in checkpoints (large)")
    parser.add_argument("--resume-dir", default="")
    parser.add_argument("--no-resume-optimizer", action="store_true")
    parser.add_argument("--note", default="MPS single-process CosyVoice3 flow finetune")
    args = parser.parse_args()

    if args.accum_grad < 1:
        raise SystemExit("--accum-grad must be >= 1")
    if args.eval_every_steps < 1 or args.save_every_steps < 1:
        raise SystemExit("--eval-every-steps and --save-every-steps must be >= 1")

    # Keep quality-oriented fp32 while letting unsupported ops run on CPU rather
    # than aborting a long MPS run.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model_dir = Path(args.model_dir).expanduser().resolve()
    checkpoint = model_dir / "flow.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing base flow checkpoint: {checkpoint}")

    train_data = Path(args.train_data).expanduser().resolve()
    cv_data = Path(args.cv_data).expanduser().resolve()
    for path in (train_data, cv_data):
        if not path.exists():
            raise SystemExit(f"missing data list: {path}")

    # Build only the flow (+ data pipeline). Null the llm/hift/hifigan blocks so
    # we neither load the 2 GB LLM nor try to construct the GAN graph.
    with open(args.config, "r", encoding="utf-8") as file:
        configs = load_hyperpyyaml(
            file,
            overrides={
                "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN"),
                "llm": None,
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
    model = configs["flow"]
    if model is None:
        raise SystemExit("config did not produce a 'flow' model")
    state = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(
        {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}, strict=False
    )
    print(f"loaded base flow: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del state

    trainable_count, total_count = set_trainable(model, args.trainable)
    model.to(device)
    model.train()
    print(f"trainable_params={trainable_count} total_params={total_count} group={args.trainable}", flush=True)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
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
        "grad_clip": args.grad_clip,
        "trainable": args.trainable,
        "trainable_modules": TRAINABLE_GROUPS[args.trainable],
        "eval_every_steps": args.eval_every_steps,
        "save_every_steps": args.save_every_steps,
        "keep_step_every": args.keep_step_every,
        "max_train_steps": args.max_train_steps,
        "max_eval_batches": args.max_eval_batches,
        "early_stop_patience": args.early_stop_patience,
        "min_delta": args.min_delta,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "note": args.note,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(out_dir / "run_metadata.json", run_metadata)
    append_jsonl(log_path, {"event": "run_start", **run_metadata, **runtime_metrics()})

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
        append_jsonl(log_path, {"event": "resumed", "resume_dir": args.resume_dir, "step": global_step})

    optimizer.zero_grad(set_to_none=True)
    should_stop = False

    try:
        for epoch in range(int(resume_state.get("epoch", -1)) + 1, args.epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            epoch_losses: list[float] = []
            pending_accum = 0
            last_train_loss: float | None = None

            for batch_idx, batch in enumerate(train_loader, 1):
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    should_stop = True
                    break

                loss_dict = model(batch, device)
                raw_loss = float(loss_dict["loss"].detach().cpu())
                (loss_dict["loss"] / args.accum_grad).backward()
                pending_accum += 1
                epoch_losses.append(raw_loss)
                last_train_loss = raw_loss

                append_jsonl(log_path, {
                    "event": "train_batch", "epoch": epoch, "batch": batch_idx, "pending_accum": pending_accum,
                    "step": global_step, "loss": raw_loss, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **runtime_metrics(),
                })

                if pending_accum >= args.accum_grad:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending_accum = 0
                    global_step += 1
                    append_jsonl(log_path, {
                        "event": "optimizer_step", "epoch": epoch, "batch": batch_idx, "step": global_step,
                        "train_loss": last_train_loss, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **runtime_metrics(),
                    })
                    tensorboard_log(writer, "train", {"loss": last_train_loss}, global_step)
                    print(f"epoch={epoch} step={global_step} loss={last_train_loss:.6f}", flush=True)

                    if global_step % args.eval_every_steps == 0:
                        old_best = best_eval_loss
                        best_eval_loss = evaluate_and_checkpoint(
                            model=model, optimizer=optimizer, cv_loader=cv_loader, device=device, args=args,
                            run_metadata={**run_metadata, "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None},
                            out_dir=out_dir, log_path=log_path, checkpoint_index=checkpoint_index, writer=writer,
                            epoch=epoch, global_step=global_step, last_train_loss=last_train_loss,
                            best_eval_loss=best_eval_loss, reason="step",
                        )
                        if best_eval_loss < old_best - args.min_delta:
                            no_improve_evals = 0
                        else:
                            no_improve_evals += 1
                        if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                            append_jsonl(log_path, {"event": "early_stop", "step": global_step, "no_improve_evals": no_improve_evals})
                            should_stop = True
                            break
                    elif global_step % args.save_every_steps == 0:
                        ckpt_metadata = {
                            **run_metadata, "event": "checkpoint_no_eval", "epoch": epoch, "step": global_step,
                            "train_loss": last_train_loss, "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **runtime_metrics(),
                        }
                        save_flow_checkpoint(model, optimizer, out_dir / "latest", ckpt_metadata, checkpoint_index, args.save_optimizer)

            if pending_accum > 0 and not should_stop:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_summary = {
                "event": "epoch_end", "epoch": epoch, "step": global_step,
                "train_loss": mean(epoch_losses), "train_batches": len(epoch_losses),
                "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
                "seconds": time.time() - epoch_start, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **runtime_metrics(),
            }
            append_jsonl(log_path, epoch_summary)
            print(json.dumps(epoch_summary, sort_keys=True), flush=True)

            if should_stop:
                break

    except KeyboardInterrupt:
        interrupted = {**run_metadata, "event": "interrupted", "step": global_step,
                       "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
                       "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        append_jsonl(log_path, interrupted)
        save_flow_checkpoint(model, optimizer, out_dir / "interrupted", interrupted, checkpoint_index, args.save_optimizer)
        if writer is not None:
            writer.close()
        raise

    done_metadata = {
        **run_metadata, "event": "run_end", "finished": True, "step": global_step,
        "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **runtime_metrics(),
    }
    save_flow_checkpoint(model, optimizer, out_dir / "final", done_metadata, checkpoint_index, args.save_optimizer)
    append_jsonl(log_path, done_metadata)
    write_json(out_dir / "final_metrics.json", done_metadata)
    if writer is not None:
        writer.close()
    print(f"flow finetune complete; checkpoints under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
