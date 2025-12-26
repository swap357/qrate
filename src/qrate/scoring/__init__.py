"""Multi-pass scoring system for exhibition-quality image selection.

Inspired by Renaissance art analysis principles (da Vinci, chiaroscuro, geometry).
How would a Renaissance master evaluate this photograph?

Scoring Passes:
    1. Technical - sharpness, exposure, noise, dynamic range
    2. Composition - thirds, balance, simplicity, negative space
    3. Color - harmony, saturation balance, contrast
    4. Renaissance - geometry, focal hierarchy, lighting, subject separation
    5. Uniqueness - distinctiveness within collection

Final exhibition score weights these to find images with
maximum artistic/monetary value for museum display.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Re-export types for convenience
from qrate.scoring.types import (
    ColorScores,
    CompositionScores,
    ExhibitionScore,
    RenaissanceScores,
    TechnicalScores,
)

# Re-export utilities
from qrate.scoring.utils import auto_orient, prepare_image

# Import pass functions for direct use
from qrate.scoring.color import (
    compute_color_contrast,
    compute_color_harmony,
    compute_color_scores,
    compute_saturation_balance,
)
from qrate.scoring.composition import (
    compute_composition_scores,
    compute_negative_space,
    compute_obstruction,
    compute_simplicity,
    compute_subject_clarity,
    compute_thirds_alignment,
    compute_visual_balance,
)
from qrate.scoring.renaissance import (
    compute_clutter_penalty,
    compute_distance_readability,
    compute_emotion_subdued,
    compute_focal_hierarchy,
    compute_geometry_strength,
    compute_highlight_clipping_penalty,
    compute_light_directional,
    compute_motion_blur_penalty,
    compute_renaissance_scores,
    compute_subject_separation,
)
from qrate.scoring.technical import (
    compute_dynamic_range,
    compute_subject_sharpness,
    compute_technical_scores,
)

__all__ = [
    # Types
    "TechnicalScores",
    "CompositionScores",
    "ColorScores",
    "RenaissanceScores",
    "ExhibitionScore",
    # Utilities
    "auto_orient",
    "prepare_image",
    # Main scoring
    "score_image",
    "score_uniqueness",
    # Pass functions
    "compute_technical_scores",
    "compute_composition_scores",
    "compute_color_scores",
    "compute_renaissance_scores",
    # Individual metrics (for direct access)
    "compute_dynamic_range",
    "compute_subject_sharpness",
    "compute_thirds_alignment",
    "compute_visual_balance",
    "compute_simplicity",
    "compute_negative_space",
    "compute_obstruction",
    "compute_subject_clarity",
    "compute_color_harmony",
    "compute_saturation_balance",
    "compute_color_contrast",
    "compute_geometry_strength",
    "compute_focal_hierarchy",
    "compute_light_directional",
    "compute_subject_separation",
    "compute_emotion_subdued",
    "compute_distance_readability",
    "compute_highlight_clipping_penalty",
    "compute_motion_blur_penalty",
    "compute_clutter_penalty",
]


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

    # Prepare image (auto-orient, convert RGB, resize)
    img = prepare_image(img)

    score = ExhibitionScore()

    # Technical pass (reuse if provided)
    if existing_technical:
        score.technical = existing_technical
    else:
        score.technical = compute_technical_scores(img)

    # Composition pass
    score.composition = compute_composition_scores(img)

    # Color pass
    score.color = compute_color_scores(img)

    # Renaissance art analysis pass
    score.renaissance = compute_renaissance_scores(img)

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
