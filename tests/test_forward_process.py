"""The forward process q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)."""

import pytest

torch = pytest.importorskip("torch")

from diffusion import build_ddpm_scheduler, to_model_range


@pytest.fixture(scope="module")
def sched():
    return build_ddpm_scheduler()


def test_add_noise_shape_and_closed_form(sched):
    g = torch.Generator().manual_seed(0)
    x0 = torch.rand(4, 3, 8, 8, generator=g) * 2 - 1
    noise = torch.randn(x0.shape, generator=g)
    t = torch.tensor([0, 10, 500, 999])

    xt = sched.add_noise(x0, noise, t)

    assert xt.shape == x0.shape
    ab = sched.alphas_cumprod[t].view(-1, 1, 1, 1)
    torch.testing.assert_close(xt, ab.sqrt() * x0 + (1 - ab).sqrt() * noise)


@pytest.mark.parametrize("t", [0, 100, 500, 900, 999])
def test_noise_variance_is_one_minus_alpha_bar(sched, t):
    # With x_0 = 0, x_t is pure scaled noise, so Var[x_t] = 1 - abar_t.
    g = torch.Generator().manual_seed(t)
    x0 = torch.zeros(64, 3, 32, 32)
    noise = torch.randn(x0.shape, generator=g)
    xt = sched.add_noise(x0, noise, torch.full((64,), t))
    expected = 1 - sched.alphas_cumprod[t].item()
    assert xt.var().item() == pytest.approx(expected, rel=0.03, abs=1e-6)


@pytest.mark.parametrize("t", [0, 250, 500, 999])
def test_process_is_variance_preserving(sched, t):
    # For unit-variance data the marginal variance stays 1 at every t.
    g = torch.Generator().manual_seed(100 + t)
    x0 = torch.randn(64, 3, 32, 32, generator=g)
    noise = torch.randn(x0.shape, generator=g)
    xt = sched.add_noise(x0, noise, torch.full((64,), t))
    assert xt.var().item() == pytest.approx(1.0, rel=0.03)


def test_endpoints_signal_vs_noise(sched):
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(16, 3, 16, 16, generator=g)
    noise = torch.randn(x0.shape, generator=g)

    def corr(a, b):
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

    x_first = sched.add_noise(x0, noise, torch.zeros(16, dtype=torch.long))
    x_last = sched.add_noise(x0, noise, torch.full((16,), 999))
    assert corr(x_first, x0) > 0.99
    assert corr(x_last, noise) > 0.99


def test_to_model_range_maps_unit_interval_to_minus_one_one():
    x = torch.tensor([0.0, 0.5, 1.0])
    torch.testing.assert_close(to_model_range(x), torch.tensor([-1.0, 0.0, 1.0]))
