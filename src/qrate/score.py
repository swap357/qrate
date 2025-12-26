"""Multi-pass scoring system for exhibition-quality image selection.

This module re-exports from qrate.scoring for backward compatibility.
New code should import directly from qrate.scoring.

Inspired by Renaissance art analysis principles (da Vinci, chiaroscuro, geometry).
How would a Renaissance master evaluate this photograph?

Scoring Passes:
    1. Geometry - strong lines, shapes, patterns (golden ratio, symmetry)
    2. Focal Hierarchy - clear visual hierarchy, subject prominence
    3. Light - directional lighting, chiaroscuro effects
    4. Subject Separation - subject clearly separated from background
    5. Emotion - subdued, contemplative (not overly dramatic)
    6. Distance - readable depth, spatial relationships
    7. Penalties - highlight clipping, motion blur, clutter

Final exhibition score weights these to find images with
maximum artistic/monetary value for museum display.
"""

from __future__ import annotations

# Re-export everything from the new modular scoring package
from qrate.scoring import (
    # Types
    ColorScores,
    CompositionScores,
    ExhibitionScore,
    RenaissanceScores,
    TechnicalScores,
    # Utilities
    auto_orient,
    # Main scoring functions
    score_image,
    score_uniqueness,
    # Technical pass
    compute_dynamic_range,
    compute_subject_sharpness,
    # Composition pass
    compute_negative_space,
    compute_obstruction,
    compute_simplicity,
    compute_subject_clarity,
    compute_thirds_alignment,
    compute_visual_balance,
    # Color pass
    compute_color_contrast,
    compute_color_harmony,
    compute_saturation_balance,
    # Renaissance pass
    compute_clutter_penalty,
    compute_distance_readability,
    compute_emotion_subdued,
    compute_focal_hierarchy,
    compute_geometry_strength,
    compute_highlight_clipping_penalty,
    compute_light_directional,
    compute_motion_blur_penalty,
    compute_subject_separation,
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
    # Main scoring
    "score_image",
    "score_uniqueness",
    # Technical pass
    "compute_dynamic_range",
    "compute_subject_sharpness",
    # Composition pass
    "compute_thirds_alignment",
    "compute_visual_balance",
    "compute_simplicity",
    "compute_negative_space",
    "compute_obstruction",
    "compute_subject_clarity",
    # Color pass
    "compute_color_harmony",
    "compute_saturation_balance",
    "compute_color_contrast",
    # Renaissance pass
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
