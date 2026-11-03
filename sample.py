"""Sample from a trained DDPM checkpoint, comparing DDPM vs DDIM samplers
with an eta ablation.

Needs a checkpoint from train.py (checkpoints are not committed to the repo).

Outputs:
  - <out-dir>/scheduler_comparison.png

Example:
  python sample.py --checkpoint checkpoints/ddpm_cifar10_final.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only ever writes PNGs

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from diffusion import (  # noqa: E402
    build_ddim_scheduler,
    build_ddpm_scheduler,
    build_model,
    resolve_device,
    sample,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare DDPM and DDIM samplers on a checkpoint.")
    p.add_argument("--checkpoint", default="checkpoints/ddpm_cifar10_final.pt")
    p.add_argument(
        "--device", default="auto", help="auto (default), cuda, mps or cpu. auto picks the GPU."
    )
    p.add_argument("--n-samples", type=int, default=8, help="Images per sampler row.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="outputs",
        help="Where the comparison grid is written. Kept separate from samples/, "
        "which holds the committed results.",
    )
    return p.parse_args(argv)


def load_model(checkpoint, device):
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        sys.exit(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train one first, e.g. `python train.py` (full run) or "
            "`python train.py --epochs 1 --max-batches 20` (smoke test)."
        )
    model = build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def grid(samples_list, labels, path, title):
    n_rows = len(samples_list)
    n_cols = samples_list[0].shape[0]
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.5 * n_cols + 2, 1.5 * n_rows), squeeze=False
    )
    for r, (samp, label) in enumerate(zip(samples_list, labels, strict=True)):
        for c in range(n_cols):
            img = (samp[c].clamp(-1, 1) + 1) / 2
            axes[r, c].imshow(img.permute(1, 2, 0).cpu().numpy())
            axes[r, c].axis("off")
        axes[r, 0].text(
            -0.18,
            0.5,
            label,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)

    ddpm_sched = build_ddpm_scheduler()
    ddim_sched = build_ddim_scheduler(ddpm_sched)
    n, seed = args.n_samples, args.seed

    s_ddpm, t_ddpm = timed(sample, model, ddpm_sched, 1000, n, device, seed=seed)
    print(f"DDPM 1000-step:    {t_ddpm:.1f}s")

    s_100, t_100 = timed(sample, model, ddim_sched, 100, n, device, seed=seed, eta=0.0)
    s_eta, t_eta = timed(sample, model, ddim_sched, 100, n, device, seed=seed, eta=0.5)
    s_50, t_50 = timed(sample, model, ddim_sched, 50, n, device, seed=seed, eta=0.0)
    print(f"DDIM 100  eta=0.0: {t_100:.1f}s")
    print(f"DDIM 100  eta=0.5: {t_eta:.1f}s")
    print(f"DDIM  50  eta=0.0: {t_50:.1f}s")

    out_path = out_dir / "scheduler_comparison.png"
    grid(
        [s_ddpm, s_eta, s_100, s_50],
        [
            f"DDPM 1000\n{t_ddpm:.1f}s",
            f"DDIM 100\neta=0.5\n{t_eta:.1f}s",
            f"DDIM 100\neta=0\n{t_100:.1f}s",
            f"DDIM  50\neta=0\n{t_50:.1f}s",
        ],
        out_path,
        "Sampler comparison — trajectory (eta) matters more than step count\n"
        "for an undertrained noise predictor",
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
