"""Properties of the cosine noise schedule used for training and sampling."""

import math

import pytest

torch = pytest.importorskip("torch")

from diffusers import DDPMScheduler

from diffusion import NUM_TRAIN_TIMESTEPS, build_ddim_scheduler, build_ddpm_scheduler


def cosine_alpha_bar(t):
    """alpha_bar(t) from Nichol & Dhariwal (2021) with s = 0.008, t in [0, 1]."""
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


@pytest.fixture(scope="module")
def sched():
    return build_ddpm_scheduler()


def test_schedule_length_and_type(sched):
    assert sched.config.num_train_timesteps == NUM_TRAIN_TIMESTEPS == 1000
    assert sched.config.beta_schedule == "squaredcos_cap_v2"
    assert sched.betas.shape == (NUM_TRAIN_TIMESTEPS,)
    assert sched.alphas_cumprod.shape == (NUM_TRAIN_TIMESTEPS,)


def test_betas_are_valid_variances(sched):
    assert torch.all(sched.betas > 0)
    assert torch.all(sched.betas <= 0.999 + 1e-6)  # the "cap" in squaredcos_cap_v2


def test_alpha_bar_decreases_from_one_to_zero(sched):
    ab = sched.alphas_cumprod
    assert torch.all(ab[1:] < ab[:-1])
    assert ab[0] > 0.999  # almost all signal at t = 0
    assert ab[-1] < 1e-3  # almost pure noise at t = T - 1


def test_alpha_bar_matches_cosine_formula(sched):
    # alphas_cumprod[i] = alpha_bar((i + 1) / T) / alpha_bar(0) until beta hits its cap
    T = NUM_TRAIN_TIMESTEPS
    expected = torch.tensor(
        [cosine_alpha_bar((i + 1) / T) / cosine_alpha_bar(0) for i in range(900)],
        dtype=torch.float32,
    )
    torch.testing.assert_close(sched.alphas_cumprod[:900], expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("t", [100, 250, 500, 750])
def test_cosine_keeps_more_signal_than_linear(sched, t):
    linear = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, beta_schedule="linear")
    assert sched.alphas_cumprod[t] > linear.alphas_cumprod[t]


def test_ddim_shares_the_ddpm_schedule(sched):
    ddim = build_ddim_scheduler(sched)
    torch.testing.assert_close(ddim.alphas_cumprod, sched.alphas_cumprod)
