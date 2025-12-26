"""Color scoring pass: harmony, saturation balance, contrast."""

from __future__ import annotations

import numpy as np
from PIL import Image

from qrate.scoring.types import ColorScores


def compute_color_harmony(img: Image.Image) -> float:
    """Score color harmony using color wheel relationships.

    Harmonious palettes: complementary, analogous, triadic.
    """
    # Convert to HSV
    hsv = img.convert("RGB")
    pixels = np.array(hsv).reshape(-1, 3)

    # Convert RGB to HSV manually (PIL HSV is 8-bit, we want float)
    r, g, b = pixels[:, 0] / 255.0, pixels[:, 1] / 255.0, pixels[:, 2] / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue calculation
    hue = np.zeros_like(delta)
    mask = delta > 0.01

    # Red is max
    red_max = mask & (cmax == r)
    hue[red_max] = 60 * (((g[red_max] - b[red_max]) / delta[red_max]) % 6)

    # Green is max
    green_max = mask & (cmax == g)
    hue[green_max] = 60 * ((b[green_max] - r[green_max]) / delta[green_max] + 2)

    # Blue is max
    blue_max = mask & (cmax == b)
    hue[blue_max] = 60 * ((r[blue_max] - g[blue_max]) / delta[blue_max] + 4)

    # Saturation (only consider saturated pixels)
    sat = np.where(cmax > 0.01, delta / cmax, 0)
    saturated = sat > 0.2  # Ignore desaturated pixels

    if saturated.sum() < 100:
        return 0.5  # Not enough color to judge

    hues = hue[saturated]

    # Build hue histogram (12 bins = 30 degrees each)
    hist, _ = np.histogram(hues, bins=12, range=(0, 360))
    hist = hist / hist.sum()

    # Find dominant hues (top 3 bins)
    top_bins = np.argsort(hist)[-3:]
    top_hues = top_bins * 30 + 15  # Bin centers

    # Check for harmony patterns
    # Complementary: ~180 degrees apart
    # Analogous: within 60 degrees
    # Triadic: ~120 degrees apart

    if len(top_hues) >= 2:
        h1, h2 = top_hues[0], top_hues[1]
        diff = abs(h1 - h2)
        if diff > 180:
            diff = 360 - diff

        # Score based on proximity to ideal relationships
        complementary_score = 1.0 - abs(diff - 180) / 90  # Peak at 180 degrees
        analogous_score = 1.0 - min(diff, 60) / 60  # Peak at 0-30 degrees
        triadic_score = 1.0 - abs(diff - 120) / 60  # Peak at 120 degrees

        return max(complementary_score, analogous_score, triadic_score, 0)

    return 0.5


def compute_saturation_balance(img: Image.Image) -> float:
    """Score saturation balance.

    Neither oversaturated (garish) nor undersaturated (muddy).
    """
    hsv = np.array(img.convert("HSV"))
    sat = hsv[:, :, 1] / 255.0

    mean_sat = sat.mean()
    std_sat = sat.std()

    # Ideal: moderate saturation (0.3-0.6) with some variation
    if 0.25 <= mean_sat <= 0.65:
        sat_score = 1.0
    else:
        # Penalty for extreme saturation
        sat_score = 1.0 - min(abs(mean_sat - 0.45) / 0.45, 1.0)

    # Some variation is good (flat = boring)
    var_score = min(1.0, std_sat / 0.2)

    return sat_score * 0.7 + var_score * 0.3


def compute_color_contrast(img: Image.Image) -> float:
    """Score color contrast/separation.

    Good images have distinct color regions, not muddy blends.
    """
    # Downsample for speed
    small = img.resize((100, 100), Image.Resampling.LANCZOS)
    pixels = np.array(small).reshape(-1, 3).astype(np.float64)

    # Simple k-means-like clustering to find dominant colors
    # Use 5 clusters
    n_colors = 5

    # Initialize with spread across color space
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    centers = pixels[rng.choice(len(pixels), n_colors, replace=False)]

    for _ in range(10):  # Simple k-means iterations
        # Assign pixels to nearest center
        dists = np.sqrt(((pixels[:, None] - centers[None, :]) ** 2).sum(axis=2))
        labels = dists.argmin(axis=1)

        # Update centers
        for i in range(n_colors):
            mask = labels == i
            if mask.sum() > 0:
                centers[i] = pixels[mask].mean(axis=0)

    # Score: distance between cluster centers
    center_dists = []
    for i in range(n_colors):
        for j in range(i + 1, n_colors):
            d = np.sqrt(((centers[i] - centers[j]) ** 2).sum())
            center_dists.append(d)

    if center_dists:
        # Max possible distance is ~441 (0,0,0 to 255,255,255)
        avg_dist = np.mean(center_dists) / 441.0
        return min(1.0, avg_dist * 2)  # Scale up

    return 0.5


def compute_color_scores(img: Image.Image) -> ColorScores:
    """Compute all color scores for an image.

    Args:
        img: PIL Image (should be RGB, pre-processed).

    Returns:
        ColorScores with all metrics computed.
    """
    return ColorScores(
        harmony=compute_color_harmony(img),
        saturation_balance=compute_saturation_balance(img),
        color_contrast=compute_color_contrast(img),
    )
