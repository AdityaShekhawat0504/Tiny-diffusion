"""Train a DDPM on CIFAR-10 using Hugging Face Diffusers.

Defaults reproduce the run in notebooks/exploration.ipynb
(10 epochs, batch 128, AdamW lr 1e-4, seed 42).

Outputs:
  - <checkpoint-dir>/ddpm_cifar10_final.pt
  - <out-dir>/loss_curve.png

Examples:
  python train.py                               # full run (~25 min on a T4 GPU)
  python train.py --epochs 1 --max-batches 20   # quick smoke test on CPU
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only ever writes PNGs

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from diffusion import (  # noqa: E402
    build_ddpm_scheduler,
    build_model,
    diffusion_loss,
    resolve_device,
    to_model_range,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a DDPM on CIFAR-10.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="auto", help="auto (default), cuda, mps or cpu. auto picks the GPU."
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop each epoch after this many batches (for quick smoke tests).",
    )
    p.add_argument("--data-dir", default="data", help="Where CIFAR-10 is downloaded.")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument(
        "--out-dir",
        default="outputs",
        help="Where the loss curve is written. Kept separate from samples/, "
        "which holds the committed results.",
    )
    return p.parse_args(argv)


def save_loss_curve(losses, path):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training loss — DDPM on CIFAR-10")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(to_model_range)])
    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f}M")

    scheduler = build_ddpm_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    t_start = time.time()
    for epoch in range(args.epochs):
        epoch_losses = []
        desc = f"Epoch {epoch + 1}/{args.epochs}"
        for i, (batch, _) in enumerate(tqdm(loader, desc=desc, leave=False)):
            if args.max_batches is not None and i >= args.max_batches:
                break
            batch = batch.to(device)
            loss = diffusion_loss(model, scheduler, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg = float(np.mean(epoch_losses))
        losses.append(avg)
        elapsed = (time.time() - t_start) / 60
        print(f"{desc}: loss = {avg:.4f}  |  elapsed: {elapsed:.1f} min")

    save_loss_curve(losses, out_dir / "loss_curve.png")

    ckpt_path = ckpt_dir / "ddpm_cifar10_final.pt"
    torch.save(
        {"model_state": model.state_dict(), "epoch": args.epochs, "losses": losses},
        ckpt_path,
    )
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Loss curve: {out_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
