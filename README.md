# Tiny Diffusion — DDPM/DDIM on CIFAR-10

I trained a DDPM from scratch on CIFAR-10 using Hugging Face Diffusers, then ran a sampler ablation comparing DDPM against DDIM at various step counts and noise injection levels (eta). The model is a vanilla UNet2DModel (~35M parameters) trained for 10 epochs — enough to produce recognisable structure, not enough for sharp semantics. The non-obvious finding: trajectory choice (eta) matters more than step count when the noise predictor is undertrained.

*Code: [train.py](train.py) · [sample.py](sample.py) · [exploration notebook](notebooks/exploration.ipynb)*

---

## Results

![Sampler comparison](samples/scheduler_comparison.png)

| Sampler | Steps | Eta | Time (8 samples) | Notes |
|---------|-------|-----|------------------|-------|
| DDPM | 1000 | — | 19.0s | Sharpest; recognisable object structure |
| DDIM | 100 | 0.5 | 4.8s | DDPM-comparable quality at 4× speedup |
| DDIM | 100 | 0.0 | 1.8s | Visibly degraded — blurry, less object structure |
| DDIM | 50 | 0.0 | 1.0s | Strongly degraded |

Deterministic DDIM (eta=0) amplifies undertraining errors because it takes large, committed noise-prediction steps with no correction mechanism. Stochastic DDIM (eta=0.5) injects per-step Gaussian noise, which provides regularisation similar to DDPM's full Markov chain and largely papers over the noise predictor's inaccuracies. With more training the eta=0 quality gap should shrink — at convergence, a perfect noise predictor doesn't need the stochasticity crutch.

---

## Training

![Training loss](samples/loss_curve.png)

I trained for 10 epochs with batch size 128, AdamW at lr 1e-4, on a Colab T4 GPU — total wall-clock time was 25 minutes. Loss decreased monotonically from 0.1470 (epoch 1) to 0.0629 (epoch 10) with no instabilities or spikes. That said, 10 epochs is substantially undertrained relative to published CIFAR-10 DDPMs, which typically use ~800k gradient steps; the model here sees roughly 3900 steps total.

---

## Sample quality

![DDPM samples](samples/ddpm_1000steps.png)

The DDPM 1000-step samples are recognisable as natural-image-like: object silhouettes (vehicles, animals) are visible in most samples, and global colour statistics look plausible. What's missing is semantic detail — boundaries are soft, fine textures are absent, and it's rarely possible to name the specific object in a sample. This is consistent with 10-epoch undertraining; the model has learned the coarse structure of the data distribution but not the high-frequency details.

---

## Implementation notes

### Noise schedule

I used the cosine schedule (`squaredcos_cap_v2`) rather than linear because linear schedules over-noise low-resolution images at early timesteps, destroying almost all signal before the network has a chance to see meaningful structure. The cosine schedule preserves more signal at low t, which is particularly important at 32×32 resolution.

### Loss objective

The training loss is MSE between the network's predicted noise and the true noise added at each timestep — the ε-prediction objective from Ho et al. (2020). This is equivalent to denoising score matching up to a timestep-dependent scaling factor.

### What I'd try next

- **Longer training (100+ epochs)** to test whether the eta=0 quality gap actually closes as the noise predictor improves
- **Class-conditional generation** with classifier-free guidance — CIFAR-10 has 10 clean classes, so this is a natural next step
- **Proper FID evaluation** using `clean-fid` against the real CIFAR-10 test set, to replace the current visual-inspection quality claims
- **v-prediction objective** instead of ε-prediction, which is reported to give better sample quality at low step counts

---

## Run it yourself

```bash
pip install -r requirements.txt
python train.py        # ~25 min on Colab T4 GPU; CPU is much slower
python sample.py
```

---

## Stack

*Python 3.12 · PyTorch · Hugging Face Diffusers · torchvision · matplotlib · NumPy*

---

## References

- Ho et al. (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Song et al. (2020), [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- Nichol & Dhariwal (2021), [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — cosine noise schedule
- [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers)
