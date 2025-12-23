"""Multi-pass scoring system for exhibition-quality image selection.

Inspired by technical photography analysis and aesthetic theory from
fine art, museum curation, and visual composition principles.

Scoring Passes:
    1. Technical - sharpness, exposure, noise, dynamic range
    2. Composition - rule of thirds, balance, simplicity, negative space
    3. Color - harmony, saturation, contrast
    4. Uniqueness - distinctiveness within the collection

Final exhibition score weights these to find images with
maximum artistic/monetary value for museum display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def auto_orient(img: Image.Image) -> Image.Image:
    """Auto-rotate image based on EXIF orientation tag.

    Critical preprocessing step - all position-based passes
    (foreground detection, subject location, rule of thirds)
    require correct orientation to work properly.

    Returns correctly oriented image (or original if no EXIF).
    """
    try:
        exif = img._getexif()
        if exif is None:
            return img

        # Find orientation tag (tag 274)
        orientation = None
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "Orientation":
                orientation = value
                break

        if orientation is None:
            return img

        # Apply rotation/flip based on EXIF orientation
        # See: https://exiftool.org/TagNames/EXIF.html
        if orientation == 2:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            return img.rotate(180, expand=True)
        elif orientation == 4:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 6:
            return img.rotate(270, expand=True)
        elif orientation == 7:
            return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 8:
            return img.rotate(90, expand=True)
        else:
            return img  # orientation == 1 or unknown
    except (AttributeError, KeyError, IndexError):
        return img


@dataclass
class TechnicalScores:
    """Technical quality metrics."""

    sharpness: float = 0.0  # Laplacian variance (higher = sharper)
    exposure: float = 0.0  # 0-1, 1 = well exposed
    noise: float = 0.0  # 0-1, 0 = clean
    dynamic_range: float = 0.0  # 0-1, 1 = full range used
    subject_sharpness: float = 0.0  # Sharpness weighted by saliency (0-1)


@dataclass
class CompositionScores:
    """Compositional/aesthetic metrics."""

    thirds_alignment: float = 0.0  # 0-1, subject on power points
    balance: float = 0.0  # 0-1, visual weight distribution
    simplicity: float = 0.0  # 0-1, low clutter
    negative_space: float = 0.0  # 0-1, effective use of empty areas
    obstruction: float = 0.0  # 0-1, 1 = no foreground obstruction
    subject_clarity: float = 0.0  # 0-1, main subject unobstructed


@dataclass
class ColorScores:
    """Color aesthetic metrics."""

    harmony: float = 0.0  # 0-1, complementary/analogous colors
    saturation_balance: float = 0.0  # 0-1, not over/under saturated
    color_contrast: float = 0.0  # 0-1, effective color separation


@dataclass
class ExhibitionScore:
    """Combined exhibition-worthiness score."""

    technical: TechnicalScores = field(default_factory=TechnicalScores)
    composition: CompositionScores = field(default_factory=CompositionScores)
    color: ColorScores = field(default_factory=ColorScores)
    uniqueness: float = 0.0  # 0-1, distinctiveness in collection

    # Weights for final score (museum curation priorities)
    WEIGHT_TECHNICAL: float = 0.30  # Technical excellence is table stakes
    WEIGHT_COMPOSITION: float = 0.35  # Composition is king in fine art
    WEIGHT_COLOR: float = 0.20  # Color harmony elevates work
    WEIGHT_UNIQUENESS: float = 0.15  # Unique perspectives are valued

    @property
    def technical_score(self) -> float:
        """Aggregate technical score (0-1)."""
        t = self.technical
        # Use subject_sharpness (saliency-weighted) for exhibition scoring
        # Falls back to raw sharpness if subject_sharpness not computed
        if t.subject_sharpness > 0:
            sharp_score = t.subject_sharpness  # Already normalized 0-1
        else:
            sharp_score = min(1.0, t.sharpness / 2000.0)
        # Noise is a penalty
        noise_penalty = 1.0 - t.noise
        return (
            sharp_score * 0.4
            + t.exposure * 0.3
            + noise_penalty * 0.15
            + t.dynamic_range * 0.15
        )

    @property
    def composition_score(self) -> float:
        """Aggregate composition score (0-1)."""
        c = self.composition
        return (
            c.thirds_alignment * 0.20
            + c.balance * 0.15
            + c.simplicity * 0.15
            + c.negative_space * 0.15
            + c.obstruction * 0.20  # Heavy weight - obstructions kill images
            + c.subject_clarity * 0.15
        )

    @property
    def color_score(self) -> float:
        """Aggregate color score (0-1)."""
        col = self.color
        return (
            col.harmony * 0.4 + col.saturation_balance * 0.3 + col.color_contrast * 0.3
        )

    @property
    def final_score(self) -> float:
        """Final exhibition score (0-100).

        When obstructions are detected, composition matters more.
        A technically perfect shot ruined by a sign is worthless for exhibition.
        """
        # Base weights
        w_tech = self.WEIGHT_TECHNICAL
        w_comp = self.WEIGHT_COMPOSITION

        # If obstruction detected, shift weight from technical to composition
        # Rationale: sharp signs don't make good art
        obstruction_penalty = (
            1.0 - self.composition.obstruction
        )  # 0 = clean, higher = obstructed
        if obstruction_penalty > 0.15:  # Significant obstruction
            # Reduce tech weight, boost composition weight
            shift = obstruction_penalty * 0.15  # Up to 15% shift
            w_tech = max(0.15, w_tech - shift)
            w_comp = min(0.45, w_comp + shift)

        raw = (
            self.technical_score * w_tech
            + self.composition_score * w_comp
            + self.color_score * self.WEIGHT_COLOR
            + self.uniqueness * self.WEIGHT_UNIQUENESS
        )
        return raw * 100


# =============================================================================
# Technical Pass
# =============================================================================


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


# =============================================================================
# Composition Pass
# =============================================================================


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
    from scipy.ndimage import uniform_filter  # type: ignore[import-untyped]

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

    from scipy.ndimage import gaussian_filter, label  # type: ignore[import-untyped]

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

    from scipy.ndimage import uniform_filter  # type: ignore[import-untyped]

    # Local contrast as saliency proxy
    local_mean = uniform_filter(gray, size=32)
    local_sqr = uniform_filter(gray**2, size=32)
    local_std = np.sqrt(np.maximum(0, local_sqr - local_mean**2))

    # Combine edge and contrast for saliency
    saliency = edges * 0.5 + local_std * 0.5
    saliency = saliency / (saliency.max() + 1e-6)

    # Find where the main "subject" is (peak saliency region)
    # Smooth to find general area
    from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

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


# =============================================================================
# Color Pass
# =============================================================================


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

    # Build hue histogram (12 bins = 30° each)
    hist, _ = np.histogram(hues, bins=12, range=(0, 360))
    hist = hist / hist.sum()

    # Find dominant hues (top 3 bins)
    top_bins = np.argsort(hist)[-3:]
    top_hues = top_bins * 30 + 15  # Bin centers

    # Check for harmony patterns
    # Complementary: ~180° apart
    # Analogous: within 60°
    # Triadic: ~120° apart

    if len(top_hues) >= 2:
        h1, h2 = top_hues[0], top_hues[1]
        diff = abs(h1 - h2)
        if diff > 180:
            diff = 360 - diff

        # Score based on proximity to ideal relationships
        complementary_score = 1.0 - abs(diff - 180) / 90  # Peak at 180°
        analogous_score = 1.0 - min(diff, 60) / 60  # Peak at 0-30°
        triadic_score = 1.0 - abs(diff - 120) / 60  # Peak at 120°

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
    centers = pixels[np.random.choice(len(pixels), n_colors, replace=False)]

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


# =============================================================================
# Main Scoring Function
# =============================================================================


def score_image(
    img: Image.Image | Path | str,
    existing_technical: TechnicalScores | None = None,
) -> ExhibitionScore:
    """Compute full exhibition score for an image.

    Args:
        img: PIL Image or path to image.
        existing_technical: Reuse existing technical scores if available.

    Returns:
        ExhibitionScore with all passes computed.
    """
    if isinstance(img, (str, Path)):
        img = Image.open(img)

    # Auto-orient based on EXIF (critical for position-based passes)
    img = auto_orient(img)

    # Ensure RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize for faster processing (keep aspect ratio, max 1024)
    max_dim = max(img.size)
    if max_dim > 1024:
        scale = 1024 / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    score = ExhibitionScore()

    # Technical pass (reuse if provided)
    if existing_technical:
        score.technical = existing_technical
    else:
        from qrate.analyze import (
            compute_exposure_score,
            compute_sharpness,
            estimate_noise,
        )

        score.technical = TechnicalScores(
            sharpness=compute_sharpness(img),
            exposure=compute_exposure_score(img),
            noise=estimate_noise(img),
            dynamic_range=compute_dynamic_range(img),
            subject_sharpness=compute_subject_sharpness(img),
        )

    # Composition pass
    score.composition = CompositionScores(
        thirds_alignment=compute_thirds_alignment(img),
        balance=compute_visual_balance(img),
        simplicity=compute_simplicity(img),
        negative_space=compute_negative_space(img),
        obstruction=compute_obstruction(img),
        subject_clarity=compute_subject_clarity(img),
    )

    # Color pass
    score.color = ColorScores(
        harmony=compute_color_harmony(img),
        saturation_balance=compute_saturation_balance(img),
        color_contrast=compute_color_contrast(img),
    )

    return score


def score_uniqueness(
    target_hash: str,
    all_hashes: list[str],
    threshold: int = 16,
) -> float:
    """Score how unique an image is within a collection.

    Args:
        target_hash: Perceptual hash of target image.
        all_hashes: All perceptual hashes in collection.
        threshold: Max Hamming distance to consider "similar".

    Returns:
        Uniqueness score 0-1 (1 = very unique).
    """
    from qrate.analyze import hamming_distance

    if not all_hashes or not target_hash:
        return 0.5

    # Find minimum distance to any other image
    min_dist = float("inf")
    similar_count = 0

    for h in all_hashes:
        if h == target_hash:
            continue
        dist = hamming_distance(target_hash, h)
        min_dist = min(min_dist, dist)
        if dist <= threshold:
            similar_count += 1

    # Score based on distance from nearest neighbor
    # and number of similar images
    if min_dist == float("inf"):
        return 1.0

    # Distance score (0 at 0, 1 at threshold*2)
    dist_score = min(1.0, min_dist / (threshold * 2))

    # Penalty for many similar images
    similarity_penalty = min(0.3, similar_count * 0.05)

    return max(0, dist_score - similarity_penalty)
