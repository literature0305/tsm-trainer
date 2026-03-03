"""Embedding-space augmentation for low-sample neural classification.

All augmentations operate on pre-extracted MantisV2 embedding tensors,
NOT on raw sensor data.  This avoids re-running the frozen backbone
and keeps augmentation in the fast, low-dimensional embedding space.

Strategies (original):
  1. GaussianNoise: additive N(0, sigma) noise
  2. Mixup: convex combination of embedding pairs + soft labels
  3. ChannelDrop: zero-out entire channel-level embedding segments

Strategies (Phase 2 v2 — evidence-based for N~100 regime):
  4. DistributionCalibration: per-class Gaussian sampling with Ledoit-Wolf
     (Yang et al., "Free Lunch for Few-Shot Learning", ICLR 2021 Oral)
  5. SMOTE: k-NN interpolation within same-class embeddings
  6. FroFA: channel-wise affine perturbation on frozen embeddings
     (FroFA, CVPR 2024, Google DeepMind)
  7. AdaptiveNoise: per-dimension noise scaled by training set std
  8. WithinClassMixup: mixup restricted to same-class pairs

DC/SMOTE operate on numpy arrays (pre-training, applied once per fold).
FroFA/AdaptiveNoise/WithinClassMixup operate on tensors (per-epoch).
"""

from __future__ import annotations

import numpy as np
import torch


# ============================================================================
# Original augmentations (tensor-space, per-epoch)
# ============================================================================

def gaussian_noise(
    Z: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to embeddings.

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Input embeddings.
    sigma : float
        Standard deviation of the noise. Values 0.01-0.1 typical.
    generator : torch.Generator, optional
        For reproducible noise.

    Returns
    -------
    Tensor, shape (batch, embed_dim)
        Noisy embeddings (original Z is not modified).
    """
    if sigma <= 0:
        return Z
    noise = torch.randn(Z.shape, dtype=Z.dtype, device=Z.device, generator=generator)
    return Z + sigma * noise


def mixup(
    Z: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup augmentation: convex combination of embedding pairs.

    Generates virtual training samples by linearly interpolating between
    random pairs.  Returns soft label vectors (one-hot mixed).

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Input embeddings.
    y : Tensor, shape (batch,) of int64 or (batch, n_classes) of float
        Labels -- integer class indices or already-soft probability vectors.
    alpha : float
        Beta distribution parameter. alpha=0 means no mixing, alpha=1 is
        uniform mixing.  Typical: 0.2-1.0.
    generator : torch.Generator, optional
        For reproducible shuffling and lambda sampling.

    Returns
    -------
    Z_mixed : Tensor, shape (batch, embed_dim)
    y_mixed : Tensor, shape (batch, n_classes) -- soft labels
    """
    if alpha <= 0:
        # No mixing -- return one-hot soft labels if integer labels provided
        if y.ndim == 1:
            n_classes = int(y.max().item()) + 1
            y_onehot = torch.zeros(len(y), n_classes, dtype=Z.dtype, device=Z.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            return Z, y_onehot
        return Z, y

    batch_size = Z.shape[0]

    # Convert integer labels to one-hot if needed
    if y.ndim == 1:
        n_classes = int(y.max().item()) + 1
        y_onehot = torch.zeros(batch_size, n_classes, dtype=Z.dtype, device=Z.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    else:
        y_onehot = y.clone()

    # Sample lambda from Beta(alpha, alpha)
    # PyTorch doesn't have Beta.sample with generator, so use numpy-style
    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample().item()

    # Random permutation for pairing
    if generator is not None:
        indices = torch.randperm(batch_size, generator=generator, device=Z.device)
    else:
        indices = torch.randperm(batch_size, device=Z.device)

    Z_mixed = lam * Z + (1 - lam) * Z[indices]
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[indices]

    return Z_mixed, y_mixed


def channel_drop(
    Z: torch.Tensor,
    n_channels: int,
    embed_per_channel: int,
    p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly zero-out full channel embedding segments.

    For multi-channel MantisV2 inputs, the embedding is a concatenation
    of per-channel embeddings.  This drops entire channel blocks with
    probability ``p``, forcing the head to not rely on any single sensor.

    Parameters
    ----------
    Z : Tensor, shape (batch, n_channels * embed_per_channel)
        Concatenated per-channel embeddings.
    n_channels : int
        Number of sensor channels.
    embed_per_channel : int
        Embedding dimension per channel.
    p : float
        Probability of dropping each channel block. Typical: 0.1-0.3.
    generator : torch.Generator, optional
        For reproducible drops.

    Returns
    -------
    Tensor, shape (batch, n_channels * embed_per_channel)
        Augmented embeddings (dropped channels zeroed).
    """
    if p <= 0 or n_channels <= 1:
        return Z

    batch_size = Z.shape[0]
    # Channel drop mask: (batch, n_channels)
    mask = torch.rand(
        batch_size, n_channels, dtype=Z.dtype, device=Z.device, generator=generator,
    ) >= p

    # Expand mask to embedding dimensions: (batch, n_channels * embed_per_channel)
    mask = mask.unsqueeze(-1).expand(-1, -1, embed_per_channel).reshape(batch_size, -1)

    return Z * mask


# ============================================================================
# Phase 2 v2: Pre-training augmentations (numpy-space, applied per fold)
# ============================================================================

def distribution_calibration(
    Z: np.ndarray,
    y: np.ndarray,
    n_synthetic_per_class: int = 50,
    alpha: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic embeddings via per-class Gaussian sampling.

    Based on Yang et al., "Free Lunch for Few-Shot Learning" (ICLR 2021).
    Estimates per-class Gaussian with Ledoit-Wolf shrinkage covariance,
    then samples synthetic points from N(mu_c, alpha * Sigma_c).

    CRITICAL: Must be called within a CV fold using ONLY training data.
    Never include the held-out test sample in distribution estimation.

    Parameters
    ----------
    Z : ndarray, shape (n_train, embed_dim)
        Training fold embeddings.
    y : ndarray, shape (n_train,)
        Training fold labels.
    n_synthetic_per_class : int
        Number of synthetic samples to generate per class.
    alpha : float
        Covariance scaling factor. Smaller = tighter around mean.
        Typical: 0.3-0.7.
    rng : np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    Z_aug : ndarray, shape (n_train + n_synthetic_total, embed_dim)
        Original + synthetic embeddings.
    y_aug : ndarray, shape (n_train + n_synthetic_total,)
        Original + synthetic labels.
    """
    if rng is None:
        rng = np.random.default_rng()

    classes = np.unique(y)
    d = Z.shape[1]
    synth_Z_list = []
    synth_y_list = []

    for c in classes:
        Z_c = Z[y == c]
        n_c = len(Z_c)

        if n_c < 2:
            # Too few samples to estimate covariance; skip this class
            continue

        mu_c = Z_c.mean(axis=0)

        # Ledoit-Wolf shrinkage covariance (pure numpy, no sklearn dependency)
        # Shrinkage target: scaled identity (trace(S)/d * I)
        Z_centered = Z_c - mu_c
        S = Z_centered.T @ Z_centered / (n_c - 1)  # sample covariance

        # Optimal shrinkage coefficient (Ledoit-Wolf 2004 formula)
        trace_S = np.trace(S)
        trace_S2 = np.sum(S * S)

        # Estimate sum of squared off-diagonal cross-products
        # Simplified: shrinkage toward diagonal with empirical intensity
        X2 = Z_centered ** 2
        sum_var_sij = (
            np.sum((Z_centered.T @ Z_centered) ** 2) / (n_c - 1) ** 2
            - trace_S2 / (n_c - 1)
        ) if n_c > 2 else 0.0

        target = (trace_S / d) * np.eye(d)
        num = sum_var_sij
        denom = trace_S2 - trace_S ** 2 / d
        if denom > 0 and n_c > 2:
            shrinkage = min(1.0, max(0.0, num / denom))
        else:
            shrinkage = 1.0  # Full shrinkage if can't estimate

        Sigma_c = (1 - shrinkage) * S + shrinkage * target

        # Scale covariance by alpha
        Sigma_scaled = alpha * Sigma_c

        # Sample from N(mu_c, alpha * Sigma_c)
        try:
            L = np.linalg.cholesky(Sigma_scaled + 1e-6 * np.eye(d))
            noise = rng.standard_normal((n_synthetic_per_class, d))
            synthetic = mu_c + noise @ L.T
        except np.linalg.LinAlgError:
            # Fallback: diagonal covariance if Cholesky fails
            std = np.sqrt(np.maximum(alpha * np.diag(S), 1e-8))
            synthetic = mu_c + rng.standard_normal((n_synthetic_per_class, d)) * std

        synth_Z_list.append(synthetic.astype(Z.dtype))
        synth_y_list.append(np.full(n_synthetic_per_class, c, dtype=y.dtype))

    if synth_Z_list:
        Z_aug = np.concatenate([Z] + synth_Z_list, axis=0)
        y_aug = np.concatenate([y] + synth_y_list, axis=0)
    else:
        Z_aug, y_aug = Z.copy(), y.copy()

    return Z_aug, y_aug


def smote_oversample(
    Z: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    n_synthetic_per_class: int = 30,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE oversampling: k-NN interpolation between same-class embeddings.

    Pure numpy implementation (no imblearn dependency).

    CRITICAL: Must be called within a CV fold using ONLY training data.

    Parameters
    ----------
    Z : ndarray, shape (n_train, embed_dim)
        Training fold embeddings.
    y : ndarray, shape (n_train,)
        Training fold labels.
    k_neighbors : int
        Number of nearest neighbors for interpolation.
    n_synthetic_per_class : int
        Number of synthetic samples to generate per class.
    rng : np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    Z_aug : ndarray, shape (n_train + n_synthetic_total, embed_dim)
    y_aug : ndarray, shape (n_train + n_synthetic_total,)
    """
    if rng is None:
        rng = np.random.default_rng()

    classes = np.unique(y)
    synth_Z_list = []
    synth_y_list = []

    for c in classes:
        Z_c = Z[y == c]
        n_c = len(Z_c)

        if n_c < 2:
            continue

        # Effective k: can't have more neighbors than samples - 1
        k_eff = min(k_neighbors, n_c - 1)

        # Memory-safe k-NN using tree-based search.
        # The previous implementation built a full (n_c, n_c, d) broadcast
        # tensor for pairwise distances, which requires O(n^2 * d) memory
        # — e.g. 162 GB for n_c=5286, d=1536 — causing OOM kills.
        # sklearn NearestNeighbors uses O(n*d) memory via KD-tree/Ball-tree.
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(
            n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean",
        )
        nn.fit(Z_c)
        _, nn_all = nn.kneighbors(Z_c)  # (n_c, k_eff+1), col 0 is self
        nn_indices = nn_all[:, 1:]  # (n_c, k_eff), exclude self

        for _ in range(n_synthetic_per_class):
            # Pick a random anchor
            anchor_idx = rng.integers(0, n_c)
            # Pick a random neighbor
            nn_idx = nn_indices[anchor_idx, rng.integers(0, k_eff)]
            # Interpolate
            lam = rng.uniform(0, 1)
            synthetic = Z_c[anchor_idx] + lam * (Z_c[nn_idx] - Z_c[anchor_idx])
            synth_Z_list.append(synthetic)
            synth_y_list.append(c)

    if synth_Z_list:
        synth_Z = np.stack(synth_Z_list).astype(Z.dtype)
        synth_y = np.array(synth_y_list, dtype=y.dtype)
        Z_aug = np.concatenate([Z, synth_Z], axis=0)
        y_aug = np.concatenate([y, synth_y], axis=0)
    else:
        Z_aug, y_aug = Z.copy(), y.copy()

    return Z_aug, y_aug


# ============================================================================
# Phase 2 v2: Per-epoch augmentations (tensor-space)
# ============================================================================

def frofa_augmentation(
    Z: torch.Tensor,
    strength: float = 0.1,
    generator: torch.Generator | None = None,
    Z_train_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """FroFA: channel-wise affine perturbation on frozen embeddings.

    Based on FroFA (CVPR 2024, Google DeepMind). Applies per-dimension
    affine transform: z_aug = alpha * z + beta, where
    alpha ~ U(1-s, 1+s) and beta ~ N(0, s * std_per_dim).

    Designed explicitly for frozen backbone + classification head regime.

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Input embeddings.
    strength : float
        Perturbation strength. Typical: 0.05-0.2.
    generator : torch.Generator, optional
    Z_train_std : Tensor, optional, shape (embed_dim,) or (1, embed_dim)
        Per-dimension std from the training fold. CRITICAL for TTA where
        batch_size=1 (LOOCV test set): Z.std(dim=0) with Bessel's
        correction yields NaN on a single sample.

    Returns
    -------
    Tensor, shape (batch, embed_dim)
    """
    if strength <= 0:
        return Z

    d = Z.shape[1]

    # Per-dimension affine: brightness (alpha) and contrast (beta)
    alpha = 1.0 + (2 * torch.rand(1, d, dtype=Z.dtype, device=Z.device, generator=generator) - 1) * strength

    # Use training std when provided (required for TTA with batch_size=1)
    if Z_train_std is not None:
        std_per_dim = Z_train_std.reshape(1, -1).to(Z.device).clamp(min=1e-8)
    else:
        std_per_dim = Z.std(dim=0, keepdim=True).clamp(min=1e-8)

    beta = torch.randn(1, d, dtype=Z.dtype, device=Z.device, generator=generator) * strength * std_per_dim

    return alpha * Z + beta


def adaptive_noise(
    Z: torch.Tensor,
    Z_train_std: torch.Tensor,
    scale: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-dimension noise scaled by training set standard deviation.

    Replaces fixed-sigma Gaussian noise with variance-aware perturbation.
    z_aug = z + scale * std_per_dim * epsilon, where epsilon ~ N(0, 1).

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Input embeddings.
    Z_train_std : Tensor, shape (embed_dim,) or (1, embed_dim)
        Per-dimension std from training fold (compute ONCE per fold).
    scale : float
        Noise scale factor. Typical: 0.05-0.2.
    generator : torch.Generator, optional

    Returns
    -------
    Tensor, shape (batch, embed_dim)
    """
    if scale <= 0:
        return Z

    std = Z_train_std.reshape(1, -1).clamp(min=1e-8)
    eps = torch.randn(Z.shape, dtype=Z.dtype, device=Z.device, generator=generator)
    return Z + scale * std * eps


def within_class_mixup(
    Z: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.3,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup restricted to same-class pairs only.

    Unlike standard mixup which mixes across classes (producing ambiguous
    soft labels), this only interpolates between same-class embeddings.
    Hard labels are preserved.

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Input embeddings.
    y : Tensor, shape (batch,) int64
        Integer class labels.
    alpha : float
        Beta distribution parameter for lambda sampling.
    generator : torch.Generator, optional

    Returns
    -------
    Z_mixed : Tensor, shape (batch, embed_dim)
    y : Tensor, shape (batch,) -- unchanged hard labels
    """
    if alpha <= 0:
        return Z, y

    batch_size = Z.shape[0]
    Z_mixed = Z.clone()

    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample().item()

    for c in y.unique():
        mask = (y == c)
        n_c = mask.sum().item()
        if n_c < 2:
            continue

        Z_c = Z[mask]
        # Random permutation within class
        if generator is not None:
            perm = torch.randperm(n_c, generator=generator, device=Z.device)
        else:
            perm = torch.randperm(n_c, device=Z.device)

        Z_mixed[mask] = lam * Z_c + (1 - lam) * Z_c[perm]

    return Z_mixed, y


# ============================================================================
# Dispatchers
# ============================================================================

def apply_pretrain_augmentation(
    Z: np.ndarray,
    y: np.ndarray,
    config: dict,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pre-training augmentation (DC or SMOTE) to numpy embeddings.

    Called ONCE per CV fold before tensor conversion. Operates on numpy
    arrays and returns augmented arrays with additional synthetic samples.

    Parameters
    ----------
    Z : ndarray, shape (n_train, embed_dim)
    y : ndarray, shape (n_train,)
    config : dict
        Must contain 'strategy' key: 'dc' or 'smote'.
        DC params: alpha, n_synthetic_per_class
        SMOTE params: k_neighbors, n_synthetic_per_class
    rng : np.random.Generator, optional

    Returns
    -------
    Z_aug, y_aug : augmented numpy arrays
    """
    if config is None:
        return Z, y

    strategy = config.get("strategy", "")

    if strategy == "dc":
        return distribution_calibration(
            Z, y,
            n_synthetic_per_class=config.get("n_synthetic", 50),
            alpha=config.get("alpha", 0.5),
            rng=rng,
        )
    elif strategy == "smote":
        return smote_oversample(
            Z, y,
            k_neighbors=config.get("k", 5),
            n_synthetic_per_class=config.get("n_synthetic", 30),
            rng=rng,
        )
    else:
        return Z, y


def apply_augmentation(
    Z: torch.Tensor,
    y: torch.Tensor,
    config: dict,
    generator: torch.Generator | None = None,
    Z_train_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply configured augmentations to training embeddings.

    Supports both original (gaussian_noise, mixup, channel_drop) and
    Phase 2 v2 (frofa, adaptive_noise, within_class_mixup) strategies.

    The 'strategy' key selects Phase 2 v2 augmentations. If absent,
    falls back to original key-based dispatch for backward compatibility.

    Parameters
    ----------
    Z : Tensor, shape (batch, embed_dim)
        Training embeddings.
    y : Tensor, shape (batch,) or (batch, n_classes)
        Training labels (integer or soft).
    config : dict
        Augmentation config. Phase 2 v2 uses 'strategy' key:
        - strategy='frofa': strength param
        - strategy='adaptive_noise': scale param (requires Z_train_std)
        - strategy='within_class_mixup': alpha param
        Legacy keys also supported:
        - gaussian_noise_sigma, mixup_alpha, channel_drop_p
    generator : torch.Generator, optional
    Z_train_std : Tensor, optional
        Per-dimension std from training fold (for adaptive_noise).

    Returns
    -------
    Z_aug, y_aug
    """
    strategy = config.get("strategy", "")

    # Phase 2 v2 strategies
    if strategy == "frofa":
        Z = frofa_augmentation(Z, strength=config.get("strength", 0.1), generator=generator)
        return Z, y

    elif strategy == "adaptive_noise":
        if Z_train_std is None:
            Z_train_std = Z.std(dim=0)
        Z = adaptive_noise(Z, Z_train_std, scale=config.get("scale", 0.1), generator=generator)
        return Z, y

    elif strategy == "within_class_mixup":
        return within_class_mixup(Z, y, alpha=config.get("alpha", 0.3), generator=generator)

    # Legacy / original augmentations (backward compatible)
    # Channel drop (preserves labels)
    ch_drop_p = config.get("channel_drop_p", 0.0)
    if ch_drop_p > 0:
        n_ch = config.get("n_channels", 1)
        embed_per_ch = config.get("embed_per_channel", Z.shape[1])
        Z = channel_drop(Z, n_ch, embed_per_ch, ch_drop_p, generator)

    # Gaussian noise (preserves labels)
    sigma = config.get("gaussian_noise_sigma", 0.0)
    if sigma > 0:
        Z = gaussian_noise(Z, sigma, generator)

    # Mixup (produces soft labels)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    if mixup_alpha > 0:
        Z, y = mixup(Z, y, mixup_alpha, generator)

    return Z, y
