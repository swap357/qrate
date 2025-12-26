"""Renaissance art analysis pass: geometry, focal hierarchy, lighting, etc.

Inspired by Renaissance art principles (da Vinci, chiaroscuro, geometry).
How would a Renaissance master evaluate this photograph?
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore[import-untyped]

from qrate.scoring.composition import compute_simplicity, compute_subject_clarity
from qrate.scoring.types import RenaissanceScores


def compute_geometry_strength(img: Image.Image) -> float:
    """Compute geometric strength: strong lines, shapes, patterns.

    Renaissance art emphasizes geometry (golden ratio, symmetry, strong lines).
    Detects edge density, line orientation consistency, and geometric patterns.

    Returns 0-1 score.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Edge detection (Sobel)
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    # Edge density (strong edges = geometric structure)
    # Use higher threshold for stronger edges
    edge_threshold = np.percentile(edges, 85)
    edge_density = (edges > edge_threshold).mean()

    # Line orientation consistency (dominant directions)
    angles = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    strong_edges = edges > edge_threshold
    angles = angles[strong_edges]

    if len(angles) > 100:
        # Histogram of orientations
        hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
        # Score based on how concentrated orientations are
        # Good geometry has 2-4 dominant directions
        hist_sorted = np.sort(hist)[::-1]
        top_dirs = hist_sorted[:4].sum()
        concentration = min(1.0, top_dirs / len(angles) * 1.5)  # Scale up
    else:
        concentration = 0

    # Symmetry score (vertical and horizontal)
    mid_w, mid_h = w // 2, h // 2
    left = gray[:, :mid_w]
    right = np.fliplr(gray[:, -mid_w:])
    top = gray[:mid_h, :]
    bottom = np.flipud(gray[-mid_h:, :])

    sym_v = (
        1.0 - np.abs(left - right).mean() / 255.0 if left.shape == right.shape else 0
    )
    sym_h = (
        1.0 - np.abs(top - bottom).mean() / 255.0 if top.shape == bottom.shape else 0
    )
    symmetry = (sym_v + sym_h) / 2.0

    # Rectilinearity (straight lines, right angles)
    # Check for horizontal and vertical edges
    horizontal_edges = np.abs(sobel_y) > np.percentile(np.abs(sobel_y), 75)
    vertical_edges = np.abs(sobel_x) > np.percentile(np.abs(sobel_x), 75)
    rectilinearity = (horizontal_edges.sum() + vertical_edges.sum()) / (h * w * 2)

    # Combine: edge density, orientation consistency, symmetry, rectilinearity
    return min(
        1.0,
        (
            edge_density * 0.3
            + concentration * 0.3
            + symmetry * 0.2
            + rectilinearity * 0.2
        ),
    )


def compute_focal_hierarchy(img: Image.Image) -> float:
    """Compute focal hierarchy: clear visual hierarchy, subject prominence.

    Renaissance art has clear focal points. Measures contrast, saliency,
    and how well the subject stands out from the background.

    Returns 0-1 score.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Contrast (high contrast = clear hierarchy)
    contrast = min(1.0, gray.std() / 100.0)  # Normalize: good contrast is ~50-100 std

    # Saliency (subject prominence) - use variance of local means
    local_mean = ndimage.uniform_filter(gray, size=20)
    saliency = min(1.0, local_mean.std() / 50.0)  # Normalize saliency

    # Center-weighted brightness (subject in center should stand out)
    center_region = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    periphery = np.concatenate(
        [
            gray[: h // 4, :].flatten(),
            gray[3 * h // 4 :, :].flatten(),
            gray[h // 4 : 3 * h // 4, : w // 4].flatten(),
            gray[h // 4 : 3 * h // 4, 3 * w // 4 :].flatten(),
        ]
    )

    if len(periphery) > 0:
        center_brightness = center_region.mean() / 255.0
        periphery_brightness = periphery.mean() / 255.0
        focal_separation = abs(center_brightness - periphery_brightness)
    else:
        focal_separation = 0.5

    return min(1.0, (contrast * 0.35 + saliency * 0.35 + focal_separation * 0.30))


def compute_light_directional(img: Image.Image) -> float:
    """Compute directional lighting: chiaroscuro effects.

    Renaissance art uses directional light (chiaroscuro). Measures
    light gradients, shadow/highlight separation, and directional patterns.

    Returns 0-1 score.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)

    # Compute gradients (directional light creates gradients)
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Strong gradients indicate directional lighting
    # Normalize: good directional light has gradients ~50-150
    gradient_strength = min(1.0, gradient_magnitude.mean() / 100.0)

    # Shadow/highlight separation (chiaroscuro)
    # Look for bimodal distribution (shadows and highlights)
    dark_region = (
        gray[gray < np.percentile(gray, 20)].mean()
        if (gray < np.percentile(gray, 20)).any()
        else 0
    )
    light_region = (
        gray[gray > np.percentile(gray, 80)].mean()
        if (gray > np.percentile(gray, 80)).any()
        else 255
    )
    separation = abs(light_region - dark_region) / 255.0

    # Directional consistency (light from one direction)
    # Check if gradients have consistent direction
    strong_gradients = gradient_magnitude > np.percentile(gradient_magnitude, 80)
    angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angles = angles[strong_gradients]

    if len(angles) > 100:
        hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
        # Good directional light has 1-2 dominant directions
        hist_sorted = np.sort(hist)[::-1]
        top_dirs = hist_sorted[:2].sum()
        directionality = min(1.0, top_dirs / len(angles) * 2.0)  # Scale up
    else:
        directionality = 0.5  # Neutral if can't determine

    # Light falloff (directional light creates falloff across image)
    # Measure variance in brightness across image
    brightness_variance = gray.std() / 255.0

    return min(
        1.0,
        (
            gradient_strength * 0.3
            + separation * 0.4
            + directionality * 0.2
            + brightness_variance * 0.1
        ),
    )


def compute_subject_separation(img: Image.Image) -> float:
    """Compute subject separation: subject clearly separated from background.

    Renaissance art separates subject from background. Measures depth cues,
    edge contrast, and subject/background distinction.

    Returns 0-1 score.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)

    # Edge strength at subject boundaries
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    # Strong edges indicate separation
    edge_strength = edges.mean() / 255.0

    # Contrast between center (subject) and periphery (background)
    h, w = gray.shape
    center_region = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    periphery = np.concatenate(
        [
            gray[: h // 4, :].flatten(),
            gray[3 * h // 4 :, :].flatten(),
            gray[h // 4 : 3 * h // 4, : w // 4].flatten(),
            gray[h // 4 : 3 * h // 4, 3 * w // 4 :].flatten(),
        ]
    )

    if len(periphery) > 0:
        center_mean = center_region.mean()
        periphery_mean = periphery.mean()
        separation = abs(center_mean - periphery_mean) / 255.0
    else:
        separation = 0

    # Use existing subject_clarity as part of separation
    clarity = compute_subject_clarity(img)

    return edge_strength * 0.3 + separation * 0.4 + clarity * 0.3


def compute_emotion_subdued(img: Image.Image) -> float:
    """Compute subdued emotion: contemplative, not overly dramatic.

    Renaissance art is often contemplative. Measures color saturation,
    contrast levels, and avoids overly dramatic compositions.

    Returns 0-1 score (higher = more subdued).
    """
    # Lower saturation = more subdued
    rgb = np.array(img, dtype=np.float64)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Saturation
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val
    saturation = np.where(max_val > 0, delta / (max_val + 1e-6), 0)
    avg_saturation = saturation.mean()

    # Lower saturation = more subdued (invert)
    saturation_score = 1.0 - min(1.0, avg_saturation * 2)

    # Contrast (moderate contrast = contemplative, extreme = dramatic)
    gray = np.array(img.convert("L"), dtype=np.float64)
    contrast = gray.std() / 255.0
    # Prefer moderate contrast (0.2-0.4 range)
    if 0.2 <= contrast <= 0.4:
        contrast_score = 1.0
    elif contrast < 0.2:
        contrast_score = contrast / 0.2
    else:
        contrast_score = max(0, 1.0 - (contrast - 0.4) / 0.6)

    return saturation_score * 0.6 + contrast_score * 0.4


def compute_distance_readability(img: Image.Image) -> float:
    """Compute distance readability: depth perception, spatial relationships.

    Renaissance art creates depth. Measures depth cues: perspective,
    atmospheric perspective (haze), and spatial layering.

    Returns 0-1 score.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Perspective cues: gradients from foreground to background
    # Typically: foreground darker/sharp, background lighter/softer
    foreground = gray[: h // 3, :].mean()  # Bottom third
    background = gray[2 * h // 3 :, :].mean()  # Top third

    # Depth gradient (lighter in distance = atmospheric perspective)
    depth_gradient = abs(background - foreground) / 255.0

    # Edge sharpness gradient (sharper in foreground)
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    foreground_edges = edges[: h // 3, :].mean()
    background_edges = edges[2 * h // 3 :, :].mean()

    if background_edges > 0 and foreground_edges > 0:
        # Normalize properly: ratio should be capped
        sharpness_gradient = foreground_edges / (background_edges + 1e-6)
        # Normalize: good depth has 1.2-2.0x ratio
        if sharpness_gradient < 1.0:
            sharpness_score = sharpness_gradient  # Foreground softer = bad
        else:
            sharpness_score = min(
                1.0, (sharpness_gradient - 1.0) / 1.0
            )  # 1.0-2.0 maps to 0-1
    else:
        sharpness_score = 0.5  # Neutral if can't compute

    # Spatial layering (multiple distinct depth planes)
    # Measure variance across horizontal slices
    slice_means = [gray[i * h // 10 : (i + 1) * h // 10, :].mean() for i in range(10)]
    if len(slice_means) > 1:
        layer_variance = min(1.0, np.var(slice_means) / 100.0)  # Normalize variance
    else:
        layer_variance = 0

    return min(
        1.0, (depth_gradient * 0.4 + sharpness_score * 0.3 + layer_variance * 0.3)
    )


def compute_highlight_clipping_penalty(img: Image.Image) -> float:
    """Penalty for highlight clipping (0-1, where 1 = no clipping)."""
    rgb = np.array(img, dtype=np.float64)
    # Check for pixels at maximum brightness
    max_brightness = rgb.max(axis=2)
    clipped_ratio = (max_brightness >= 254).mean()  # Near max = clipped
    return 1.0 - min(1.0, clipped_ratio * 2)  # Penalty increases with clipping


def compute_motion_blur_penalty(img: Image.Image) -> float:
    """Penalty for motion blur (0-1, where 1 = sharp)."""
    from qrate.analyze import compute_sharpness

    sharpness = compute_sharpness(img)
    # Normalize sharpness to 0-1 (assuming good sharpness is > 500)
    sharp_score = min(1.0, sharpness / 1000.0)
    return sharp_score


def compute_clutter_penalty(img: Image.Image) -> float:
    """Penalty for clutter (0-1, where 1 = clean)."""
    # Reuse existing simplicity score (inverted)
    simplicity = compute_simplicity(img)
    return simplicity


def compute_renaissance_scores(img: Image.Image) -> RenaissanceScores:
    """Compute all Renaissance art analysis scores for an image.

    Args:
        img: PIL Image (should be RGB, pre-processed).

    Returns:
        RenaissanceScores with all metrics computed.
    """
    return RenaissanceScores(
        geometry_strength=compute_geometry_strength(img),
        focal_hierarchy=compute_focal_hierarchy(img),
        light_directional=compute_light_directional(img),
        subject_separation=compute_subject_separation(img),
        emotion_subdued=compute_emotion_subdued(img),
        distance_readable=compute_distance_readability(img),
        highlight_clipping_penalty=compute_highlight_clipping_penalty(img),
        motion_blur_penalty=compute_motion_blur_penalty(img),
        clutter_penalty=compute_clutter_penalty(img),
    )
