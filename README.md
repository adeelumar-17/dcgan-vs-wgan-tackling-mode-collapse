# DCGAN vs WGAN-GP — Tackling Mode Collapse in Generative Adversarial Networks

A side-by-side PyTorch comparison of **Deep Convolutional GAN (DCGAN)** and **Wasserstein GAN with Gradient Penalty (WGAN-GP)** on the task of anime face generation, demonstrating how WGAN-GP addresses the **mode collapse** problem inherent in standard GAN training.

<img width="1461" height="295" alt="image" src="https://github.com/user-attachments/assets/c2da908b-ebae-4cc0-b3ad-432b49481424" />
**DC-GAN**
<img width="1467" height="220" alt="image" src="https://github.com/user-attachments/assets/01a325c5-6386-47a7-88df-d9513f0b02b3" />
**WGAN**
<img width="1461" height="219" alt="image" src="https://github.com/user-attachments/assets/852d1cbb-ddf2-4b87-a79d-e5703fe55963" />



| Item | Detail |
|---|---|
| **Foundation Paper** | [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) (Goodfellow et al., NeurIPS 2014) |
| **DCGAN Paper** | [Unsupervised Representation Learning with DCGANs](https://arxiv.org/abs/1511.06434) (Radford et al., ICLR 2016) |
| **WGAN-GP Paper** | [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) (Gulrajani et al., NeurIPS 2017) |
| **Dataset** | [Anime Faces](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) (~21K images) |
| **Platform** | Kaggle (GPU) |
| **Framework** | PyTorch |

---

## Table of Contents

1. [The Problem: Mode Collapse](#the-problem-mode-collapse)
2. [The Solution: From DCGAN to WGAN-GP](#the-solution-from-dcgan-to-wgan-gp)
3. [Architecture Overview](#architecture-overview)
4. [Notebook Walkthrough](#notebook-walkthrough)
5. [Training Process](#training-process)
6. [Results & Analysis](#results--analysis)
7. [How to Run](#how-to-run)
8. [Requirements](#requirements)
9. [References](#references)

---

## The Problem: Mode Collapse

The original GAN framework (Goodfellow et al., 2014) defines a minimax game between two networks:

- **Generator (G):** Maps random noise **z** to a fake image, trying to fool the discriminator.
- **Discriminator (D):** Classifies images as real or fake.

The training objective is:

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

While theoretically elegant, this formulation suffers from a critical failure mode: **mode collapse**. This occurs when the generator learns to produce only a small subset of possible outputs — sometimes collapsing to generate nearly identical images regardless of the input noise. The generator "finds a cheat" that consistently fools the discriminator, rather than learning the full data distribution.

### Why does mode collapse happen?

1. **Vanishing gradients:** When the discriminator becomes too powerful (which is easy with BCE loss), the gradient signal to the generator vanishes. The generator stops learning or learns erratically.
2. **JS divergence saturation:** The standard GAN loss is related to the Jensen-Shannon divergence. When the real and generated distributions have minimal overlap (common early in training), JS divergence saturates — it provides no useful gradient.
3. **Non-informative loss:** The discriminator loss in standard GANs doesn't meaningfully correlate with sample quality. A low D loss can coexist with terrible generations.

---

## The Solution: From DCGAN to WGAN-GP

This notebook implements **two** GAN variants to compare their behaviour:

### DCGAN (Baseline)

DCGAN (Radford et al., 2016) improves on the original GAN with architectural guidelines for stable training using convolutional networks:

- Replace pooling layers with **strided convolutions** (discriminator) and **fractional-strided convolutions** (generator).
- Use **BatchNorm** in both networks (except D's first layer and G's output layer).
- Use **ReLU** in the generator, **LeakyReLU** in the discriminator.
- Use **Tanh** activation in the generator's output layer.
- Train with **BCE loss** (standard GAN objective).

> **DCGAN still uses the standard GAN loss**, so it remains susceptible to mode collapse, vanishing gradients, and training instability.

### WGAN-GP (Improved)

WGAN-GP (Gulrajani et al., 2017) fundamentally changes the training objective by replacing the JS divergence with the **Wasserstein-1 (Earth Mover's) distance**:

```
W(P_real, P_fake) = sup_||f||_L≤1  E[f(x)] - E[f(G(z))]
```

Key changes from standard GANs:

| Aspect | DCGAN (Standard GAN) | WGAN-GP |
|---|---|---|
| **D/C role** | Discriminator (classifier) | **Critic** (scores, no classification) |
| **D/C output** | Sigmoid (probability) | **Raw score** (no sigmoid) |
| **Loss function** | `BCEWithLogitsLoss` | **Wasserstein loss:** `E[C(fake)] - E[C(real)]` |
| **Constraint** | None | **Gradient Penalty** (λ=10) |
| **Normalization** | BatchNorm in D | **No BatchNorm** in Critic |
| **Optimizer** | Adam (β₁=0.5) | Adam (**β₁=0.0**, β₂=0.9) |
| **Critic iters** | 1 D step per G step | **5** Critic steps per G step |

### Why WGAN-GP solves mode collapse

1. **Meaningful loss metric:** The Wasserstein distance provides a smooth, continuous measure of distance between distributions — even when they don't overlap. Gradients never vanish.
2. **Loss correlates with quality:** Unlike BCE-based GANs, the WGAN critic loss directly correlates with sample quality. A decreasing critic loss genuinely means the generator is improving.
3. **Gradient Penalty (GP):** Instead of weight clipping (original WGAN), GP enforces the 1-Lipschitz constraint softly by penalising the critic's gradient norm when it deviates from 1. This avoids capacity underuse from clipping and yields smoother training.
4. **More critic training:** Training the critic 5× per generator step gives the generator a stronger, more informative gradient signal — the critic accurately approximates the Wasserstein distance.

---

## Architecture Overview

Both DCGAN and WGAN-GP **share the same Generator architecture**. The difference lies in the discriminator/critic and the training procedure.

### Generator (shared)

```
z (100×1×1)
    │
    ▼
┌──────────────────────────────────┐
│ ConvTranspose2d(100→512, 4×4)   │  → 512×4×4
│ BatchNorm2d + ReLU              │
├──────────────────────────────────┤
│ ConvTranspose2d(512→256, 4×4)   │  → 256×8×8
│ BatchNorm2d + ReLU              │
├──────────────────────────────────┤
│ ConvTranspose2d(256→128, 4×4)   │  → 128×16×16
│ BatchNorm2d + ReLU              │
├──────────────────────────────────┤
│ ConvTranspose2d(128→64, 4×4)    │  → 64×32×32
│ BatchNorm2d + ReLU              │
├──────────────────────────────────┤
│ ConvTranspose2d(64→3, 4×4)      │  → 3×64×64
│ Tanh                            │
└──────────────────────────────────┘
    │
    ▼
  Fake image (3×64×64), values in [-1, 1]
```

- All convolutions use `stride=2, padding=1` (except the first: `stride=1, padding=0`).
- `bias=False` throughout (since BatchNorm handles the bias).
- **Tanh** output matches the `[-1, 1]` normalisation applied to real images.

### DCGAN Discriminator

```
Image (3×64×64)
    │
    ▼
┌──────────────────────────────────┐
│ Conv2d(3→64, 4×4, stride 2)     │  → 64×32×32
│ LeakyReLU(0.2)                  │  (no BatchNorm on first layer)
├──────────────────────────────────┤
│ Conv2d(64→128, 4×4, stride 2)   │  → 128×16×16
│ BatchNorm2d + LeakyReLU(0.2)    │
├──────────────────────────────────┤
│ Conv2d(128→256, 4×4, stride 2)  │  → 256×8×8
│ BatchNorm2d + LeakyReLU(0.2)    │
├──────────────────────────────────┤
│ Conv2d(256→512, 4×4, stride 2)  │  → 512×4×4
│ BatchNorm2d + LeakyReLU(0.2)    │
├──────────────────────────────────┤
│ Conv2d(512→1, 4×4, stride 1)    │  → 1×1×1
│ (no activation — logits for BCE) │
└──────────────────────────────────┘
    │
    ▼
  Scalar logit (real/fake probability via BCEWithLogitsLoss)
```

### WGAN-GP Critic

The Critic has the **same convolutional structure** as the Discriminator but with two critical differences:

1. **No BatchNorm** — BatchNorm interferes with the gradient penalty by introducing dependencies between samples in a batch. All `BatchNorm2d` layers are removed.
2. **No sigmoid/activation** on the output — the critic produces a **raw score** (not a probability). Higher scores = "more real".

### Model Size

| Component | Parameters |
|---|---|
| Generator | **~3.6M** |
| Discriminator (DCGAN) | **~2.8M** |
| Critic (WGAN-GP) | **~2.8M** |

### Weight Initialisation

Following the DCGAN paper, all convolutional weights are initialised from `N(0, 0.02)` and BatchNorm weights from `N(1.0, 0.02)` with biases set to 0.

---

## Notebook Walkthrough

### 1 — Imports & Device Setup

Sets up PyTorch, torchvision, and GPU detection. Seeds fixed at 42 for reproducibility.

### 2 — Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `LATENT_DIM` | **100** | Size of noise vector **z** |
| `IMG_SIZE` | **64** | 64×64 output images |
| `CHANNELS` | **3** | RGB |
| `BATCH_SIZE` | **64** | |
| `LR` | **0.0002** | DCGAN learning rate (both G and D) |
| `BETA1` | **0.5** | Adam β₁ for DCGAN |
| `BETA2` | **0.999** | Adam β₂ for DCGAN |
| `NUM_EPOCHS_DCGAN` | **60** | 50 initial + 10 extended |
| `NUM_EPOCHS_WGAN` | **50** | |
| `LAMBDA_GP` | **10** | Gradient penalty weight |
| `CRITIC_ITERS` | **5** | Critic updates per G update |

### 3 — Data Loading

Loads the **Anime Faces** dataset via `ImageFolder`. Images are resized to 64×64, centre-cropped, and normalised to `[-1, 1]` using `Normalize([0.5]*3, [0.5]*3)`.

### 4 — Real Image Visualisation

Displays a 4×4 grid of real anime faces from the dataset for reference.

### 5 — Weight Initialisation Function

`init_weights()` — applies DCGAN-prescribed weight initialisation to all Conv and BatchNorm layers.

### 6 — Generator Definition

`Generator` class — 5-layer ConvTranspose2d network mapping 100-dim noise to 3×64×64 images (see [Architecture](#generator-shared)).

### 7 — Discriminator Definition

`Discriminator` class — 5-layer Conv2d network with BatchNorm and LeakyReLU. Outputs a scalar logit per image.

### 8 — Critic Definition

`Critic` class — same architecture as Discriminator but **without BatchNorm** and **without any final activation**. Outputs a raw score.

### 9 — Gradient Penalty Function

`compute_gradient_penalty()`:
1. Samples a random interpolation factor `α ∈ [0, 1]` per sample.
2. Creates interpolated images: `x̂ = αx_real + (1-α)x_fake`.
3. Computes critic scores on interpolated images.
4. Uses `torch.autograd.grad()` to compute gradients of critic scores w.r.t. interpolated images.
5. Penalty = `E[(||∇C(x̂)||₂ - 1)²]` — pushes gradient norms toward 1 (1-Lipschitz constraint).

### 10 — Sanity Check

Instantiates Generator and Discriminator, runs a forward pass with random noise, and verifies output shapes (`[B, 3, 64, 64]` for G, `[B]` for D). Prints parameter counts.

### 11 — DCGAN Training (50 epochs)

Standard GAN training loop:
- **D step:** BCE loss on real (label-smoothed to 0.9) + BCE loss on fake (label 0.0). Uses `autocast` + `GradScaler` for mixed precision.
- **G step:** BCE loss pushing D(fake) → 1.0.
- Saves sample grids every 5 epochs and checkpoints every 10 epochs.

### 12 — DCGAN Loss Curve

Plots G and D losses across all epochs.

### 13 — DCGAN Extended Training (+10 epochs to 60)

Resumes DCGAN training for 10 additional epochs (total: 60).

### 14 — DCGAN Checkpoint Save

Saves `dcgan_checkpoint_epoch60.pth` (full state) and `dcgan_generator_final.pth` (generator weights only).

### 15 — DCGAN Extended Loss Curve

Updated loss plot including all 60 epochs.

### 16 — WGAN-GP Training (50 epochs)

WGAN-GP training loop:
- **Critic step (×5 per batch):** Wasserstein loss + gradient penalty. No mixed precision (GP requires full-precision `autograd.grad`).
- **G step:** Minimise `-E[C(G(z))]` (make critic score fakes highly).
- Saves sample grids every 5 epochs and checkpoints every 5 epochs.

### 17 — WGAN-GP Loss Curve

Plots G and Critic losses.

### 18 — Side-by-Side Comparison

Generates 16 images from the **same noise vector** through both generators, displayed as DCGAN vs WGAN-GP grids.

### 19 — Diversity Check (Mode Collapse Test)

Generates **64 samples** from each model to visually inspect diversity. DCGAN samples tend to be more repetitive; WGAN-GP samples show greater variation in hair colour, eye colour, pose, and background.

### 20 — Training Progress

Displays epoch-by-epoch sample grids for both models side-by-side, showing how quality evolves over training.

### 21 — Loss Comparison

Side-by-side loss plots: DCGAN (left) vs WGAN-GP (right), highlighting the differences in training dynamics.

### 22 — WGAN-GP Final Save

Saves `wgan_generator_final.pth` for deployment.

---

## Training Process

### DCGAN Training Details

| Aspect | Detail |
|---|---|
| **Loss** | `BCEWithLogitsLoss` — standard binary cross-entropy. |
| **Optimizer** | Adam (`lr=2e-4`, `β₁=0.5`, `β₂=0.999`) for both G and D. |
| **Label smoothing** | Real labels set to **0.9** instead of 1.0 — a common trick to prevent the discriminator from becoming overconfident. |
| **Mixed precision** | `autocast` + `GradScaler` for faster training. |
| **D:G ratio** | **1:1** — one D update per G update. |
| **Epochs** | **60** (50 + 10 extended). |

### WGAN-GP Training Details

| Aspect | Detail |
|---|---|
| **Loss** | **Wasserstein loss:** `C(fake).mean() - C(real).mean() + λ·GP`. |
| **Optimizer** | Adam (`lr=1e-4`, **`β₁=0.0`**, `β₂=0.9`) — per the WGAN-GP paper. |
| **Gradient Penalty** | λ = **10**. Enforces 1-Lipschitz constraint on the critic. |
| **Mixed precision** | **Not used** — gradient penalty requires `torch.autograd.grad` which needs full precision. |
| **Critic:G ratio** | **5:1** — five critic updates per generator update. |
| **Epochs** | **50**. |

### DCGAN Training Dynamics

- **Discriminator loss** drops quickly then oscillates, often converging to near-zero — indicating D easily distinguishes real from fake.
- **Generator loss** tends to be erratic — spikes and drops without a clear downward trend.
- This oscillation is characteristic: when D is too strong, G receives vanishing gradients and may collapse.

### WGAN-GP Training Dynamics

- **Critic loss** starts negative and gradually moves toward zero — reflecting the Wasserstein distance decreasing as G improves.
- **Generator loss** shows a steady downward trend — directly correlating with improving sample quality.
- Training is **much more stable** — no sudden spikes or oscillations.

---

## Results & Analysis

### Visual Quality

Both models produce recognisable anime faces, but with clear qualitative differences:

| Aspect | DCGAN | WGAN-GP |
|---|---|---|
| **Overall quality** | Decent faces with some artefacts | **Sharper**, more detailed faces |
| **Colour consistency** | Generally good | Generally good |
| **Facial features** | Sometimes blurry or distorted eyes | **Clearer** eye structure |
| **Background** | Often noisy | **Cleaner** backgrounds |

### Mode Collapse (The Key Comparison)

This is the central experiment of the notebook. When generating **64 samples** from random noise:

| Observation | DCGAN | WGAN-GP |
|---|---|---|
| **Diversity** | Tends toward repetitive outputs — similar hair colours, poses, and expressions | **Much greater diversity** — varied hair, eyes, poses |
| **Mode collapse** | Shows signs of partial mode collapse | **Minimal mode collapse** |
| **Face variations** | Limited variety in generated attributes | Wider range of anime character styles |

> **Key insight:** DCGAN's BCE-based training allows the generator to "cheat" by finding a few modes that reliably fool the discriminator. WGAN-GP's Wasserstein distance measures the full distributional distance, penalising mode-dropping and incentivising the generator to cover more of the data distribution.

### Loss Behaviour (Why This Matters)

| Metric | DCGAN | WGAN-GP |
|---|---|---|
| **D/C loss** | Quickly saturates near 0 | Gradually approaches 0 |
| **G loss** | Erratic, oscillating | Smooth, steadily decreasing |
| **Loss-quality correlation** | **Weak** — low D loss ≠ good images | **Strong** — decreasing C loss = better images |

> In DCGAN, you cannot tell from the loss curves alone whether training is going well. In WGAN-GP, the critic loss is a **reliable proxy for generation quality** — this is one of the most important practical advantages of the Wasserstein formulation.

### Why WGAN-GP Outperforms DCGAN

1. **Gradient Penalty > No Constraint:** GP softly enforces the Lipschitz constraint everywhere, providing smooth, informative gradients to the generator at all times.

2. **No BatchNorm in Critic:** Removing BatchNorm prevents inter-sample dependencies, ensuring the gradient penalty is computed correctly per sample.

3. **β₁ = 0:** Disabling the first moment (momentum) in Adam prevents the critic from "remembering" old gradient directions, keeping it responsive to the generator's evolving distribution.

4. **5:1 Critic-Generator Ratio:** The well-trained critic provides a much more accurate Wasserstein distance estimate, giving the generator precise gradient signals.

5. **Wasserstein Distance Properties:** Unlike JS divergence, the Wasserstein distance is continuous and differentiable even when distributions are supported on non-overlapping manifolds — the typical case early in training.

### Training Time Comparison

| Model | Epochs | Approximate Time |
|---|---|---|
| DCGAN | 60 | ~38 min |
| WGAN-GP | 50 | ~252 min (~4.2 hrs) |

> WGAN-GP is **~8× slower per epoch** than DCGAN due to: (1) 5× critic updates per generator step, (2) gradient penalty computation requiring `torch.autograd.grad`, and (3) no mixed-precision support for the GP computation. This is the computational cost of stable, mode-collapse-resistant training.

---

## How to Run

1. **Platform:** Upload the notebook to [Kaggle](https://www.kaggle.com/) and attach the [Anime Faces dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces).
2. **GPU:** Enable a GPU accelerator in the Kaggle notebook settings.
3. **Execute cells in order.** DCGAN trains first (~38 min), then WGAN-GP (~4 hrs).
4. **Comparison cells** at the end generate side-by-side outputs for direct visual comparison.

---

## Requirements

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Model definition, transforms, DataLoader, `make_grid` |
| `torch.cuda.amp` | Mixed-precision training for DCGAN (`GradScaler`, `autocast`) |
| `torch.autograd` | Gradient penalty computation for WGAN-GP |
| `numpy` | Seed setting, numerical operations |
| `matplotlib` | Visualisation (sample grids, loss curves) |
| `time`, `os` | Timing, file paths |

All dependencies are pre-installed in the default Kaggle Python 3 Docker image.

---

## References

1. **Goodfellow, I., et al.** (2014). *Generative Adversarial Nets.* NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) — The foundational GAN paper defining the adversarial training framework.

2. **Radford, A., Metz, L., & Chintala, S.** (2016). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* ICLR 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434) — DCGAN: architectural guidelines for stable convolutional GAN training.

3. **Arjovsky, M., Chintala, S., & Bottou, L.** (2017). *Wasserstein Generative Adversarial Networks.* ICML 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875) — The original WGAN paper proposing the Wasserstein distance as an alternative to JS divergence.

4. **Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A.** (2017). *Improved Training of Wasserstein GANs.* NeurIPS 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028) — WGAN-GP: replaces weight clipping with gradient penalty for better stability.

---

## License

This project is for educational purposes (Generative AI course — AI4009 Assignment 01). Feel free to use and adapt with attribution.
