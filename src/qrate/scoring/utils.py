"""Shared utilities for scoring passes."""

from __future__ import annotations

from PIL import Image


def auto_orient(img: Image.Image) -> Image.Image:
    """Auto-rotate image based on EXIF orientation tag.

    Critical preprocessing step - all position-based passes
    (foreground detection, subject location, rule of thirds)
    require correct orientation to work properly.

    Returns correctly oriented image (or original if no EXIF).
    """
    try:
        exif = img.getexif()
        if not exif:
            return img

        # Orientation is tag 274
        orientation = exif.get(274)
        if orientation is None:
            return img

        # Apply rotation/flip based on EXIF orientation
        if orientation == 2:
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            return img.rotate(180, expand=True)
        elif orientation == 4:
            return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(
                270, expand=True
            )
        elif orientation == 6:
            return img.rotate(270, expand=True)
        elif orientation == 7:
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(
                90, expand=True
            )
        elif orientation == 8:
            return img.rotate(90, expand=True)
        else:
            return img  # orientation == 1 or unknown
    except (AttributeError, KeyError, IndexError):
        return img


def prepare_image(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Prepare image for scoring: auto-orient, convert to RGB, resize.

    Args:
        img: Input PIL Image.
        max_dim: Maximum dimension (width or height) for processing.

    Returns:
        Processed image ready for scoring.
    """
    # Auto-orient based on EXIF
    img = auto_orient(img)

    # Ensure RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize for faster processing (keep aspect ratio)
    current_max = max(img.size)
    if current_max > max_dim:
        scale = max_dim / current_max
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img
