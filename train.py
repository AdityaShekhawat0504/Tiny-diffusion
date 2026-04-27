"""Train a DDPM on CIFAR-10 using HuggingFace Diffusers.

Outputs:
  - checkpoints/ddpm_cifar10_final.pt
  - samples/loss_curve.png
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-4
SEED = 42


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Path("samples").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    model = UNet2DModel(
        sample_size=32,
        in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f}M")

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    losses = []
    t_start = time.time()
    for epoch in range(EPOCHS):
        epoch_losses = []
        for batch, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch.size(0),), device=device,
            ).long()
            noisy = scheduler.add_noise(batch, noise, timesteps)
            pred = model(noisy, timesteps).sample
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg = float(np.mean(epoch_losses))
        losses.append(avg)
        elapsed = (time.time() - t_start) / 60
        print(f"Epoch {epoch+1}/{EPOCHS}: loss = {avg:.4f}  |  elapsed: {elapsed:.1f} min")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training loss — DDPM on CIFAR-10")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("samples/loss_curve.png", dpi=120, bbox_inches="tight")

    torch.save({
        "model_state": model.state_dict(),
        "epoch": EPOCHS,
        "losses": losses,
    }, "checkpoints/ddpm_cifar10_final.pt")
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
