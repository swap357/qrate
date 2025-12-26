"""Composition scoring pass: thirds, balance, simplicity, negative space."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, uniform_filter  # type: ignore[import-untyped]

from qrate.scoring.types import CompositionScores


def compute_thirds_alignment(img: Image.Image) -> float:
    """Score alignment of visual interest with rule of thirds.

    Power points are at intersections of thirds lines.
    Strong images place subjects/edges near these points.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Compute edge magnitude as proxy for visual interest
    # Simple Sobel approximation
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(gx**2 + gy**2)

    # Normalize to create "attention map"
    edges = edges / (edges.max() + 1e-6)

    # Check thirds lines (power points are at intersections)
    thirds_mask = np.zeros_like(edges)
    margin = min(h, w) // 20  # ~5% margin around lines

    # Horizontal thirds
    for y in [h // 3, 2 * h // 3]:
        thirds_mask[max(0, y - margin) : min(h, y + margin), :] = 1
    # Vertical thirds
    for x in [w // 3, 2 * w // 3]:
        thirds_mask[:, max(0, x - margin) : min(w, x + margin)] = 1

    # Score: how much edge energy is near thirds vs elsewhere
    on_thirds = (edges * thirds_mask).sum()
    total = edges.sum() + 1e-6

    return min(1.0, on_thirds / total * 2)  # Scale up since thirds area is small


def compute_visual_balance(img: Image.Image) -> float:
    """Score visual weight distribution.

    Well-balanced images have center of visual mass near center,
    or deliberately offset for dynamic tension.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Invert so bright areas have more "weight" (visual attention)
    weight = 255 - gray

    # Compute center of mass
    total = weight.sum() + 1e-6
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    cy = (y_coords * weight).sum() / total
    cx = (x_coords * weight).sum() / total

    # Distance from image center (normalized)
    center_y, center_x = h / 2, w / 2
    dist = math.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2)
    max_dist = math.sqrt(center_y**2 + center_x**2)

    # Score: centered = 1.0, edge = 0.0
    # But allow some offset for dynamic composition
    normalized_dist = dist / max_dist
    # Sweet spot is 0-0.3 from center (golden ratio area)
    if normalized_dist < 0.3:
        return 1.0
    elif normalized_dist < 0.5:
        return 1.0 - (normalized_dist - 0.3) / 0.2 * 0.3  # Gradual falloff
    else:
        return max(0, 0.7 - (normalized_dist - 0.5))


def compute_simplicity(img: Image.Image) -> float:
    """Score visual simplicity/clarity.

    Museum-quality images often have clear subjects without clutter.
    Measured via edge density and entropy.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)

    # Edge density (too many edges = busy/cluttered)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(gx**2 + gy**2)

    # Normalize edge map
    edge_density = edges.mean() / 128.0  # Typical max gradient

    # Entropy (information content - lower = simpler)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros for log
    entropy = -np.sum(hist * np.log2(hist)) / 8.0  # Normalize to 0-1

    # Simplicity is inverse of complexity
    complexity = edge_density * 0.6 + entropy * 0.4
    return max(0, 1.0 - complexity)


def compute_negative_space(img: Image.Image) -> float:
    """Score effective use of negative space.

    Strong images often have clear areas that let subjects breathe.
    Too little = cramped, too much = empty.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)

    # Find "quiet" areas (low local variance)
    local_mean = uniform_filter(gray, size=32)
    local_sqr_mean = uniform_filter(gray**2, size=32)
    local_var = local_sqr_mean - local_mean**2

    # Threshold for "quiet" regions
    quiet_threshold = 100  # Low variance = uniform area
    quiet_mask = local_var < quiet_threshold

    quiet_ratio = quiet_mask.sum() / quiet_mask.size

    # Sweet spot: 20-50% negative space is often ideal
    if 0.2 <= quiet_ratio <= 0.5:
        return 1.0
    elif quiet_ratio < 0.2:
        return quiet_ratio / 0.2  # Too busy
    else:
        return max(0, 1.0 - (quiet_ratio - 0.5) / 0.5)  # Too empty


def compute_obstruction(img: Image.Image) -> float:
    """Detect visual clutter/competing focal points.

    Exhibition-quality images have clear subjects without visual competition.
    This detects scattered visual interest (multiple high-contrast regions)
    which suggests clutter, signs, or obstructions.

    Returns 1.0 for clean focused images, lower for cluttered.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Compute edge magnitude as visual interest proxy
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(gx**2 + gy**2)

    # Smooth and find high-interest regions
    smooth_edges = gaussian_filter(edges, sigma=h // 30)
    threshold = np.percentile(smooth_edges, 80)
    high_interest = smooth_edges > threshold

    # Count distinct high-interest regions (connected components)
    labeled, num_regions = label(high_interest)

    # Good images: 1-3 major regions (subject + maybe secondary)
    # Cluttered images: many scattered regions
    if num_regions <= 2:
        region_penalty = 0
    elif num_regions <= 4:
        region_penalty = (num_regions - 2) * 0.1
    else:
        region_penalty = min(0.4, (num_regions - 2) * 0.08)

    # Also check: is there a dominant region or scattered small ones?
    if num_regions > 0:
        region_sizes = []
        for i in range(1, num_regions + 1):
            region_sizes.append((labeled == i).sum())
        region_sizes.sort(reverse=True)

        total_interest = sum(region_sizes)
        if total_interest > 0:
            # Fragmentation: if largest region is <40% of total, very scattered
            largest_ratio = region_sizes[0] / total_interest
            if largest_ratio < 0.3:
                fragmentation_penalty = 0.3
            elif largest_ratio < 0.5:
                fragmentation_penalty = (0.5 - largest_ratio) * 0.5
            else:
                fragmentation_penalty = 0
        else:
            fragmentation_penalty = 0
    else:
        fragmentation_penalty = 0

    total_penalty = region_penalty + fragmentation_penalty
    return max(0, 1.0 - total_penalty)


def compute_subject_clarity(img: Image.Image) -> float:
    """Score how clearly the main subject is visible.

    Uses saliency estimation to find the likely subject,
    then checks if it's obstructed by foreground elements.

    Strong images have clear subjects in the middle/upper frame,
    not blocked by foreground clutter.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Simple saliency: combination of edges and local contrast
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(gx**2 + gy**2)

    # Local contrast as saliency proxy
    local_mean = uniform_filter(gray, size=32)
    local_sqr = uniform_filter(gray**2, size=32)
    local_std = np.sqrt(np.maximum(0, local_sqr - local_mean**2))

    # Combine edge and contrast for saliency
    saliency = edges * 0.5 + local_std * 0.5
    saliency = saliency / (saliency.max() + 1e-6)

    # Find where the main "subject" is (peak saliency region)
    # Smooth to find general area
    smooth_saliency = gaussian_filter(saliency, sigma=h // 20)

    # Find centroid of high-saliency region
    threshold = smooth_saliency.max() * 0.5
    salient_mask = smooth_saliency > threshold

    if salient_mask.sum() == 0:
        return 0.5  # Can't determine subject

    y_coords, _ = np.mgrid[0:h, 0:w]
    subject_y = (y_coords * salient_mask).sum() / salient_mask.sum()

    # Good: subject in middle or upper portion (not blocked by foreground)
    # y position: 0 = top, h = bottom
    # Subject in lower 30% suggests it might be obstructed or is itself an obstruction
    y_ratio = subject_y / h

    if y_ratio < 0.6:
        position_score = 1.0  # Subject in good position
    elif y_ratio < 0.75:
        position_score = 1.0 - (y_ratio - 0.6) / 0.15 * 0.4  # Gradual penalty
    else:
        position_score = 0.6 - (y_ratio - 0.75) / 0.25 * 0.4  # Heavier penalty

    # Check if there's high activity between viewer and subject
    # (stuff in front of the subject)
    if subject_y > h * 0.3:  # Subject not at very top
        foreground = saliency[int(subject_y) :, :]
        background = saliency[: int(subject_y), :]

        fg_activity = foreground.mean()
        bg_activity = background.mean()

        # If foreground is busier than background, something's blocking
        if bg_activity > 0:
            blocking_ratio = fg_activity / (bg_activity + 1e-6)
            blocking_penalty = max(0, (blocking_ratio - 0.8) / 1.5)
        else:
            blocking_penalty = 0.3
    else:
        blocking_penalty = 0

    return max(0, position_score - blocking_penalty * 0.5)


def compute_composition_scores(img: Image.Image) -> CompositionScores:
    """Compute all composition scores for an image.

    Args:
        img: PIL Image (should be RGB, pre-processed).

    Returns:
        CompositionScores with all metrics computed.
    """
    return CompositionScores(
        thirds_alignment=compute_thirds_alignment(img),
        balance=compute_visual_balance(img),
        simplicity=compute_simplicity(img),
        negative_space=compute_negative_space(img),
        obstruction=compute_obstruction(img),
        subject_clarity=compute_subject_clarity(img),
    )
