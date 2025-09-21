## Denoising Autoencoder — Overview

**Goal:** remove Gaussian noise from pistachio images (kirmizi & siirt) using a convolutional autoencoder and a light U-Net variant.
**Noise:** synthetic Gaussian (mean=0.0, std=0.1) added to clean images.

**Data & file:** everything in one notebook (`Denoising Autoencoder - Machine Learning Project.ipynb`).

**Base autoencoder (summary):**

* Input: `100 × 100 × 3` image
* Encoder: `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64)` (down to 25×25×64)
* Decoder: `UpSampling → Conv2D(64) → UpSampling → Conv2D(32) → Conv2D(3)` (output 100×100×3)
* Kernels: `3×3`, activations: `ReLU` (all), final layer: `sigmoid`

**Modified model:** lightweight U-Net variant (smaller channel counts / fewer layers) used as an alternative to improve reconstruction while keeping the model small.

**Training:** `Adam` optimizer, `MSE` loss, early stopping / model checkpoint recommended.

**Evaluation:** quantitative = **SSIM** (primary metric). Qualitative = side-by-side `clean | noisy | denoised` visual checks.
