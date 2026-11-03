"""DDPM and DDIM reverse processes with a tiny UNet and a few steps."""

import pytest

torch = pytest.importorskip("torch")

from diffusion import build_ddim_scheduler, build_ddpm_scheduler, build_model, sample

TINY = {
    "sample_size": 8,
    "block_out_channels": (32, 64),
    "down_block_types": ("DownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "UpBlock2D"),
    "layers_per_block": 1,
}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(**TINY).eval()


def test_ddpm_sampler_shape(model):
    sched = build_ddpm_scheduler()
    x = sample(model, sched, n_steps=3, n_samples=2, device="cpu", progress=False)
    assert x.shape == (2, 3, 8, 8)
    assert torch.isfinite(x).all()
    assert len(sched.timesteps) == 3


@pytest.mark.parametrize("eta", [0.0, 0.5])
@pytest.mark.parametrize("n_steps", [2, 5])
def test_ddim_sampler_shape(model, eta, n_steps):
    sched = build_ddim_scheduler()
    x = sample(model, sched, n_steps=n_steps, n_samples=2, device="cpu", eta=eta, progress=False)
    assert x.shape == (2, 3, 8, 8)
    assert torch.isfinite(x).all()
    assert len(sched.timesteps) == n_steps


def test_sampling_is_reproducible_with_seed(model):
    sched = build_ddim_scheduler()
    a = sample(model, sched, 3, 2, "cpu", seed=7, eta=0.5, progress=False)
    b = sample(model, sched, 3, 2, "cpu", seed=7, eta=0.5, progress=False)
    c = sample(model, sched, 3, 2, "cpu", seed=8, eta=0.5, progress=False)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)
