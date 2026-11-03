"""Shared building blocks for training and sampling.

Everything here mirrors the code in notebooks/exploration.ipynb: the same UNet
config, the same cosine noise schedule, the same epsilon-prediction loss and
the same sampling loop. train.py, sample.py and the tests all import from here
so the three never drift apart.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm

NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE = "squaredcos_cap_v2"  # cosine schedule (Nichol & Dhariwal, 2021)

# The UNet used for every result in this repo (~24M parameters).
MODEL_CONFIG = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "block_out_channels": (64, 128, 256, 256),
    "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
}


def resolve_device(name: str = "auto") -> str:
    """Return "cuda", "mps" or "cpu". "auto" picks the best one available."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_model_range(t: torch.Tensor) -> torch.Tensor:
    """Map an image tensor from [0, 1] (ToTensor output) to [-1, 1].

    A named function instead of a lambda so DataLoader workers can pickle it
    on macOS and Windows.
    """
    return (t * 2) - 1


def build_model(**overrides) -> UNet2DModel:
    """Build the UNet. Keyword overrides are only used by the tests."""
    return UNet2DModel(**{**MODEL_CONFIG, **overrides})


def build_ddpm_scheduler() -> DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_schedule=BETA_SCHEDULE)


def build_ddim_scheduler(ddpm_scheduler: DDPMScheduler | None = None) -> DDIMScheduler:
    """DDIM sampler sharing the exact noise schedule of the DDPM scheduler."""
    ddpm_scheduler = ddpm_scheduler or build_ddpm_scheduler()
    return DDIMScheduler.from_config(ddpm_scheduler.config)


def diffusion_loss(model: UNet2DModel, scheduler: DDPMScheduler, batch: torch.Tensor):
    """Epsilon-prediction loss (Ho et al., 2020) for one batch of clean images.

    Sample t uniformly, noise the batch with q(x_t | x_0), and regress the
    added noise with MSE.
    """
    noise = torch.randn_like(batch)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch.size(0),),
        device=batch.device,
    ).long()
    noisy = scheduler.add_noise(batch, noise, timesteps)
    pred = model(noisy, timesteps).sample
    return F.mse_loss(pred, noise)


@torch.no_grad()
def sample(model, sched, n_steps, n_samples, device, seed=42, eta=None, progress=True):
    """Run the reverse process from pure noise and return images in [-1, 1] (unclamped).

    Works with DDPMScheduler and DDIMScheduler. `eta` is only passed to DDIM.
    """
    torch.manual_seed(seed)
    sched.set_timesteps(n_steps)
    size = model.config.sample_size
    x = torch.randn(n_samples, model.config.in_channels, size, size, device=device)
    for t in tqdm(sched.timesteps, leave=False, disable=not progress):
        pred_noise = model(x, t).sample
        kwargs = {"eta": eta} if eta is not None else {}
        x = sched.step(pred_noise, t, x, **kwargs).prev_sample
    return x
