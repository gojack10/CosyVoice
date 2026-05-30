#!/usr/bin/env python3
"""Single-process CosyVoice3 HiFT (vocoder) GAN finetuner.

Finetunes the HiFT generator adversarially on the target voice so the vocoder
learns VEGA's fine waveform texture (the phasey 'doubled' character) that the
mel-domain flow physically cannot represent. Reuses CosyVoice's HiFiGan loss
wrapper (adversarial + feature-match + 45x mel-recon anchor + f0).

The released model ships no discriminator, so D starts random. Mitigations:
- warm up D first (G frozen) so its gradients are sane before touching G;
- gentle G lr; the built-in 45x mel-recon weight anchors fidelity;
- frequent generator-only snapshots (base hift stays the fallback).

CPU by default (MPS hit a Metal assertion on the flow run). Generator-only
checkpoints save as `hift.pt` so synth loads them straight into model.hift.
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

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.dataset.dataset import Dataset  # noqa: E402

# Compat shim: this checkout's Dataset(gan=True) re-wraps compute_fbank with
# token_mel_ratio=0, but the installed compute_fbank signature predates that
# kwarg (num_frames already handles the 25hz alignment). Accept and ignore it,
# else GAN-pipeline iteration raises TypeError. Patch before load_hyperpyyaml so
# the yaml's !name binding picks up the tolerant version.
import functools as _ft  # noqa: E402
import cosyvoice.dataset.processor as _proc  # noqa: E402

if "token_mel_ratio" not in _proc.compute_fbank.__code__.co_varnames:
    _orig_compute_fbank = _proc.compute_fbank

    @_ft.wraps(_orig_compute_fbank)
    def _compute_fbank_compat(data, feat_extractor, num_frames=-1, mode="train", token_mel_ratio=0):
        return _orig_compute_fbank(data, feat_extractor, num_frames=num_frames, mode=mode)

    _proc.compute_fbank = _compute_fbank_compat


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_to_f32(batch: dict) -> dict:
    # MPS has no float64; pyworld's pitch_feat comes back float64. Downcast so the
    # batch can move to MPS. Harmless on CPU.
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dtype == torch.float64:
            batch[k] = v.float()
    return batch


def cpu_state(m: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def mean(v: list[float]) -> float | None:
    return float(sum(v) / len(v)) if v else None


def rss_gb() -> float:
    r = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return (r if sys.platform == "darwin" else r * 1024.0) / 1024 ** 3


def save_gen_ckpt(generator, ckpt_dir: Path, meta: dict, index: Path, optimizer=None, disc=None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state(generator), ckpt_dir / "hift.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer_g.pt")
    if disc is not None:
        torch.save(cpu_state(disc), ckpt_dir / "discriminator.pt")
    write_json(ckpt_dir / "train_state.json", meta)
    write_json(ckpt_dir / "config.json", {"format": "cosyvoice3_hift_finetune", "config": meta})
    append_jsonl(index, {"event": "checkpoint", "path": str(ckpt_dir), **meta})


def copy_ckpt(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.replace(dst)


def run_eval(model, loader, device, max_batches: int) -> float | None:
    """Generator mel-reconstruction loss on dev as a fidelity proxy."""
    model.eval()
    mels: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if max_batches > 0 and i > max_batches:
                break
            mels.append(float(model.forward_generator(batch_to_f32(batch), device)["loss_mel"].detach().cpu()))
    model.train()
    return mean(mels)


def main() -> int:
    p = argparse.ArgumentParser(description="Finetune CosyVoice3 HiFT adversarially on one process")
    p.add_argument("--model-dir", default=str(Path.home() / "projects/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    p.add_argument("--config", default=str(ROOT_DIR / "local_voice_pipeline/conf/cosyvoice3_sft.yaml"))
    p.add_argument("--train-data", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/train.data.list"))
    p.add_argument("--cv-data", default=str(ROOT_DIR / "local_voice_pipeline/data/cosyvoice3/dev.data.list"))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "local_voice_pipeline/exp/cosyvoice3/hift_gan/run1"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-train-steps", type=int, default=400, help="0 = no cap (counts batches)")
    p.add_argument("--d-warmup-steps", type=int, default=60, help="train only the discriminator for this many steps first")
    p.add_argument("--max-eval-batches", type=int, default=3)
    p.add_argument("--eval-every-steps", type=int, default=50)
    p.add_argument("--save-every-steps", type=int, default=50)
    p.add_argument("--keep-step-every", type=int, default=100, help="0 disables; else snapshot a step-tagged hift.pt")
    p.add_argument("--lr-g", type=float, default=1e-5)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--freeze-f0", action=argparse.BooleanOptionalAction, default=True, help="freeze HiFT f0_predictor; focuses the finetune on waveform texture (use --no-freeze-f0 to train it)")
    p.add_argument("--save-optimizer", action="store_true")
    p.add_argument("--note", default="single-process CosyVoice3 HiFT GAN finetune")
    a = p.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    model_dir = Path(a.model_dir).expanduser().resolve()
    hift_pt = model_dir / "hift.pt"
    if not hift_pt.exists():
        raise SystemExit(f"missing base hift: {hift_pt}")
    train_data = Path(a.train_data).expanduser().resolve()
    cv_data = Path(a.cv_data).expanduser().resolve()

    with open(a.config, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f, overrides={
            "qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN"),
            "llm": None,
            "flow": None,
        })

    train_ds = Dataset(str(train_data), data_pipeline=configs["data_pipeline_gan"], mode="train", gan=True, dpo=False, shuffle=True, partition=False)
    cv_ds = Dataset(str(cv_data), data_pipeline=configs["data_pipeline_gan"], mode="dev", gan=True, dpo=False, shuffle=False, partition=False)
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
    cv_loader = DataLoader(cv_ds, batch_size=None, num_workers=0)

    device = choose_device(a.device)
    print(f"using device={device}", flush=True)
    model = configs["hifigan"]
    if model is None:
        raise SystemExit("config did not build 'hifigan'")
    state = torch.load(hift_pt, map_location="cpu")
    miss, unexp = model.generator.load_state_dict({k: v for k, v in state.items() if isinstance(v, torch.Tensor)}, strict=False)
    print(f"loaded base hift into generator: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    del state
    model.to(device)
    model.train()

    if a.freeze_f0 and hasattr(model.generator, "f0_predictor"):
        for _p in model.generator.f0_predictor.parameters():
            _p.requires_grad_(False)
        print("froze generator.f0_predictor (NSF f0 stays at base prediction)", flush=True)
    gen_params = [p for p in model.generator.parameters() if p.requires_grad]
    disc_params = list(model.discriminator.parameters())
    opt_g = torch.optim.AdamW(gen_params, lr=a.lr_g, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc_params, lr=a.lr_d, betas=(0.8, 0.99))
    print(f"gen_params={sum(p.numel() for p in gen_params)} disc_params={sum(p.numel() for p in disc_params)}", flush=True)

    out_dir = Path(a.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    index = out_dir / "checkpoint_index.jsonl"
    meta_base = {
        "base_hift": str(hift_pt), "out_dir": str(out_dir), "device": str(device),
        "lr_g": a.lr_g, "lr_d": a.lr_d, "d_warmup_steps": a.d_warmup_steps, "grad_clip": a.grad_clip,
        "max_train_steps": a.max_train_steps, "note": a.note, "torch": torch.__version__,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(out_dir / "run_metadata.json", meta_base)
    append_jsonl(log_path, {"event": "run_start", **meta_base, "rss_gb": rss_gb()})

    global_step = 0
    best_eval = math.inf
    should_stop = False

    def do_eval_ckpt(step, epoch, last):
        nonlocal best_eval
        ev = run_eval(model, cv_loader, device, a.max_eval_batches)
        rec = {"event": "eval", "step": step, "epoch": epoch, "eval_loss_mel": ev, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rss_gb": rss_gb(), **last}
        append_jsonl(log_path, rec)
        print(json.dumps(rec, sort_keys=True), flush=True)
        meta = {**meta_base, **rec, "finished": False}
        save_gen_ckpt(model.generator, out_dir / "latest", meta, index, opt_g if a.save_optimizer else None, model.discriminator if a.save_optimizer else None)
        if a.keep_step_every > 0 and step % a.keep_step_every == 0:
            copy_ckpt(out_dir / "latest", out_dir / f"step_{step:06d}")
        if ev is not None and ev < best_eval:
            best_eval = ev
            copy_ckpt(out_dir / "latest", out_dir / "best")
            append_jsonl(log_path, {"event": "new_best", "step": step, "eval_loss_mel": ev})

    try:
        for epoch in range(a.epochs):
            train_ds.set_epoch(epoch)
            for batch in train_loader:
                if a.max_train_steps > 0 and global_step >= a.max_train_steps:
                    should_stop = True
                    break
                batch = batch_to_f32(batch)
                warming = global_step < a.d_warmup_steps

                # --- discriminator step ---
                opt_d.zero_grad(set_to_none=True)
                d_out = model.forward_discriminator(batch, device)
                d_out["loss"].backward()
                if a.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(disc_params, a.grad_clip)
                opt_d.step()

                g_log: dict[str, Any] = {}
                if not warming:
                    # --- generator step ---
                    opt_g.zero_grad(set_to_none=True)
                    opt_d.zero_grad(set_to_none=True)  # discard stray D grads from G graph
                    g_out = model.forward_generator(batch, device)
                    g_out["loss"].backward()
                    if a.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(gen_params, a.grad_clip)
                    opt_g.step()
                    g_log = {k: float(v.detach().cpu()) for k, v in g_out.items() if torch.is_tensor(v)}

                last = {"loss_disc": float(d_out["loss"].detach().cpu()), **{f"g_{k}": v for k, v in g_log.items()}}
                global_step += 1
                if global_step % 10 == 0 or warming:
                    print(f"step={global_step} warmup={warming} loss_d={last['loss_disc']:.4f}"
                          + (f" g_loss={g_log.get('loss', float('nan')):.4f} g_mel={g_log.get('loss_mel', float('nan')):.4f}" if g_log else ""), flush=True)
                append_jsonl(log_path, {"event": "step", "step": global_step, "epoch": epoch, "warmup": warming, **last, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rss_gb": rss_gb()})

                if global_step % a.eval_every_steps == 0:
                    do_eval_ckpt(global_step, epoch, last)
            if should_stop:
                break
    except KeyboardInterrupt:
        meta = {**meta_base, "event": "interrupted", "step": global_step}
        save_gen_ckpt(model.generator, out_dir / "interrupted", meta, index)
        raise

    final_meta = {**meta_base, "event": "run_end", "finished": True, "step": global_step,
                  "best_eval_loss_mel": best_eval if math.isfinite(best_eval) else None,
                  "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    save_gen_ckpt(model.generator, out_dir / "final", final_meta, index)
    append_jsonl(log_path, final_meta)
    write_json(out_dir / "final_metrics.json", final_meta)
    print(f"hift finetune complete; checkpoints under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
