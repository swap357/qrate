"""Technical scoring pass: sharpness, exposure, noise, dynamic range."""

from __future__ import annotations

import numpy as np
from PIL import Image

from qrate.scoring.types import TechnicalScores


def compute_dynamic_range(img: Image.Image) -> float:
    """Measure effective use of tonal range.

    Good images use the full histogram without crushing blacks/whites.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()

    # Find effective range (where 98% of pixels live)
    cumsum = np.cumsum(hist)
    low = np.searchsorted(cumsum, 0.01)
    high = np.searchsorted(cumsum, 0.99)

    # Score based on range width and avoiding extremes
    range_width = (high - low) / 256.0

    # Penalty for crushed blacks/whites (>5% at extremes)
    black_crush = max(0, hist[:8].sum() - 0.05)
    white_crush = max(0, hist[-8:].sum() - 0.05)

    return max(0, range_width - black_crush - white_crush)


def compute_subject_sharpness(img: Image.Image) -> float:
    """Compute sharpness weighted by saliency (subject region).

    Unlike raw sharpness which can be inflated by sharp foreground
    obstructions (signs, text), this focuses on what matters:
    sharpness where the likely subject is.

    Uses variance of Laplacian (same as analyze.compute_sharpness)
    but only on the detected subject region.

    Returns normalized 0-1 score.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Compute Laplacian (same as analyze.compute_sharpness)
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )

    # Create subject mask: upper/center regions, avoiding foreground
    # Method: combine center bias + foreground penalty

    # Center bias weight
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    center_y, center_x = h / 2, w / 2
    dist_from_center = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    center_weight = 1.0 - (dist_from_center / max_dist)

    # Foreground penalty (bottom portion often has obstructions)
    fg_weight = np.ones((h, w))
    fg_weight[int(h * 0.7) :, :] = 0.1  # Heavy penalty on bottom 30%
    fg_weight[int(h * 0.5) : int(h * 0.7), :] = 0.5  # Mild penalty

    # Combined weight for subject region
    subject_weight = center_weight * fg_weight
    subject_weight = subject_weight / (subject_weight.max() + 1e-6)

    # Use top 40% of weighted area as subject
    threshold = np.percentile(subject_weight, 60)
    subject_mask = subject_weight > threshold

    if subject_mask.sum() < 100:
        # Fallback: upper 60% of image
        subject_mask = np.zeros((h, w), dtype=bool)
        subject_mask[: int(h * 0.6), :] = True

    # Compute variance of Laplacian on subject region (same metric as original)
    subject_lap = lap[subject_mask]
    subject_variance = float(subject_lap.var())

    # Also compute foreground variance for comparison
    fg_mask = np.zeros((h, w), dtype=bool)
    fg_mask[int(h * 0.7) :, :] = True
    fg_lap = lap[fg_mask]
    fg_variance = float(fg_lap.var()) if fg_mask.sum() > 0 else 0

    # Penalty if foreground is significantly sharper than subject
    # This catches cases where signs/obstructions have sharp edges
    if subject_variance > 0 and fg_variance > subject_variance * 1.3:
        # Foreground sharper than subject - likely obstruction
        ratio = fg_variance / subject_variance
        penalty = min(0.4, (ratio - 1.3) * 0.2)
    else:
        penalty = 0

    # Normalize to 0-1 (variance typically 0-3000 for sharp images)
    normalized = min(1.0, subject_variance / 2000.0)

    return max(0, normalized - penalty)


def compute_technical_scores(img: Image.Image) -> TechnicalScores:
    """Compute all technical scores for an image.

    Args:
        img: PIL Image (should be RGB, pre-processed).

    Returns:
        TechnicalScores with all metrics computed.
    """
    from qrate.analyze import (
        compute_exposure_score,
        compute_sharpness,
        estimate_noise,
    )

    return TechnicalScores(
        sharpness=compute_sharpness(img),
        exposure=compute_exposure_score(img),
        noise=estimate_noise(img),
        dynamic_range=compute_dynamic_range(img),
        subject_sharpness=compute_subject_sharpness(img),
    )
