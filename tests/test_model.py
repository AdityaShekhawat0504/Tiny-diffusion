"""The UNet noise predictor and the training loss."""

import pytest

torch = pytest.importorskip("torch")

from diffusion import MODEL_CONFIG, build_ddpm_scheduler, build_model, diffusion_loss

# A small UNet with the same block types, so CPU tests stay fast.
TINY = {
    "sample_size": 8,
    "block_out_channels": (32, 64),
    "down_block_types": ("DownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "UpBlock2D"),
    "layers_per_block": 1,
}


def test_full_model_size_and_forward_shape():
    torch.manual_seed(0)
    model = build_model().eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert 23.5e6 < n_params < 24.5e6  # notebook reports 23.99M

    size = MODEL_CONFIG["sample_size"]
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        out = model(x, torch.tensor([0, 999])).sample
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_model_accepts_scalar_timestep():
    torch.manual_seed(0)
    model = build_model(**TINY).eval()
    x = torch.randn(3, 3, 8, 8)
    with torch.no_grad():
        out = model(x, torch.tensor(500)).sample
    assert out.shape == x.shape


def test_diffusion_loss_is_finite_scalar_and_trains():
    torch.manual_seed(0)
    model = build_model(**TINY)
    sched = build_ddpm_scheduler()
    batch = torch.rand(4, 3, 8, 8) * 2 - 1

    loss = diffusion_loss(model, sched, batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients reached the model"
    assert all(torch.isfinite(g).all() for g in grads)
