"""Sample from a trained DDPM checkpoint, comparing DDPM vs DDIM samplers
with an eta ablation.

Outputs:
  - samples/scheduler_comparison.png
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm


def build_model(device):
    model = UNet2DModel(
        sample_size=32,
        in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    ckpt = torch.load("checkpoints/ddpm_cifar10_final.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def sample(model, sched, n_steps, n_samples, device, seed=42, eta=None):
    torch.manual_seed(seed)
    sched.set_timesteps(n_steps)
    x = torch.randn(n_samples, 3, 32, 32, device=device)
    for t in tqdm(sched.timesteps, leave=False):
        pred_noise = model(x, t).sample
        kwargs = {"eta": eta} if eta is not None else {}
        x = sched.step(pred_noise, t, x, **kwargs).prev_sample
    return x


def grid(samples_list, labels, path, title):
    n_rows = len(samples_list)
    n_cols = samples_list[0].shape[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols + 2, 1.5 * n_rows))
    for r, (samp, label) in enumerate(zip(samples_list, labels)):
        for c in range(n_cols):
            img = (samp[c].clamp(-1, 1) + 1) / 2
            axes[r, c].imshow(img.permute(1, 2, 0).cpu().numpy())
            axes[r, c].axis("off")
        axes[r, 0].text(
            -0.18, 0.5, label,
            transform=axes[r, 0].transAxes,
            ha="right", va="center", fontsize=10,
        )
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    Path("samples").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)

    ddpm_sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    ddim_sched = DDIMScheduler.from_config(ddpm_sched.config)

    t0 = time.time(); s_ddpm = sample(model, ddpm_sched, 1000, 8, device); t_ddpm = time.time() - t0
    print(f"DDPM 1000-step:    {t_ddpm:.1f}s")

    t0 = time.time(); s_100 = sample(model, ddim_sched, 100, 8, device, eta=0.0); t_100 = time.time() - t0
    t0 = time.time(); s_eta = sample(model, ddim_sched, 100, 8, device, eta=0.5); t_eta = time.time() - t0
    t0 = time.time(); s_50  = sample(model, ddim_sched, 50,  8, device, eta=0.0); t_50  = time.time() - t0
    print(f"DDIM 100  eta=0.0: {t_100:.1f}s")
    print(f"DDIM 100  eta=0.5: {t_eta:.1f}s")
    print(f"DDIM  50  eta=0.0: {t_50:.1f}s")

    grid(
        [s_ddpm, s_eta, s_100, s_50],
        [f"DDPM 1000\n{t_ddpm:.1f}s",
         f"DDIM 100\neta=0.5\n{t_eta:.1f}s",
         f"DDIM 100\neta=0\n{t_100:.1f}s",
         f"DDIM  50\neta=0\n{t_50:.1f}s"],
        "samples/scheduler_comparison.png",
        "Sampler comparison — trajectory (eta) matters more than step count\n"
        "for an undertrained noise predictor",
    )


if __name__ == "__main__":
    main()
