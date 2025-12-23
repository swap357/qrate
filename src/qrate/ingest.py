"""Ingest module: EXIF extraction and preview generation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

# Import rawpy only when needed (expensive import)
_rawpy = None


def _get_rawpy():  # type: ignore[no-untyped-def]
    """Lazy import rawpy."""
    global _rawpy
    if _rawpy is None:
        import rawpy  # type: ignore[import-untyped]

        _rawpy = rawpy
    return _rawpy


@dataclass
class ExifData:
    """Extracted EXIF metadata."""

    timestamp: datetime | None = None
    iso: int | None = None
    shutter: float | None = None  # seconds
    aperture: float | None = None  # f-number
    focal_length: float | None = None  # mm
    camera_make: str | None = None
    camera_model: str | None = None


def extract_exif(path: Path) -> ExifData:
    """Extract EXIF data from a RAW file using rawpy/libraw.

    Args:
        path: Path to RAW file.

    Returns:
        ExifData with available metadata.
    """
    rawpy = _get_rawpy()

    try:
        with rawpy.imread(str(path)) as raw:
            # rawpy doesn't expose full EXIF directly, we get what we can
            # Use PIL on the thumbnail for basic metadata
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
                exif = img._getexif() if hasattr(img, "_getexif") else None
                if exif:
                    return _parse_pil_exif(exif)

        # Fallback: try PIL directly on the file (works for DNG, some others)
        return _extract_exif_pil(path)
    except Exception:
        # If rawpy fails, try PIL directly
        return _extract_exif_pil(path)


def _extract_exif_pil(path: Path) -> ExifData:
    """Extract EXIF using PIL (works for DNG, some RAWs with embedded preview)."""
    try:
        with Image.open(path) as img:
            exif = img._getexif() if hasattr(img, "_getexif") else None
            if exif:
                return _parse_pil_exif(exif)
    except Exception:
        pass
    return ExifData()


def _parse_pil_exif(exif: dict) -> ExifData:
    """Parse PIL EXIF dict into ExifData."""
    # EXIF tag IDs
    TAG_DATETIME_ORIGINAL = 36867
    TAG_ISO = 34855
    TAG_EXPOSURE_TIME = 33434
    TAG_FNUMBER = 33437
    TAG_FOCAL_LENGTH = 37386
    TAG_MAKE = 271
    TAG_MODEL = 272

    result = ExifData()

    # Parse timestamp
    dt_str = exif.get(TAG_DATETIME_ORIGINAL)
    if dt_str:
        try:
            result.timestamp = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass

    # ISO
    iso = exif.get(TAG_ISO)
    if iso:
        result.iso = int(iso) if not isinstance(iso, tuple) else int(iso[0])

    # Shutter speed (exposure time in seconds)
    exp = exif.get(TAG_EXPOSURE_TIME)
    if exp:
        result.shutter = (
            float(exp)
            if not hasattr(exp, "numerator")
            else exp.numerator / exp.denominator
        )

    # Aperture (f-number)
    fnum = exif.get(TAG_FNUMBER)
    if fnum:
        result.aperture = (
            float(fnum)
            if not hasattr(fnum, "numerator")
            else fnum.numerator / fnum.denominator
        )

    # Focal length
    fl = exif.get(TAG_FOCAL_LENGTH)
    if fl:
        result.focal_length = (
            float(fl) if not hasattr(fl, "numerator") else fl.numerator / fl.denominator
        )

    # Camera info
    result.camera_make = exif.get(TAG_MAKE)
    result.camera_model = exif.get(TAG_MODEL)

    return result


def extract_preview(path: Path, output_dir: Path, max_size: int = 1024) -> Path | None:
    """Extract preview JPEG from RAW file.

    Args:
        path: Path to RAW file.
        output_dir: Directory to save preview.
        max_size: Maximum dimension (width or height) for preview.

    Returns:
        Path to preview JPEG, or None if extraction failed.
    """
    rawpy = _get_rawpy()

    # Create preview filename based on original
    preview_name = f"{path.stem}_preview.jpg"
    preview_path = output_dir / preview_name

    # Skip if already exists
    if preview_path.exists():
        return preview_path

    try:
        with rawpy.imread(str(path)) as raw:
            # Try to extract embedded thumbnail first (fastest)
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    pil_img: Image.Image = Image.open(io.BytesIO(thumb.data))
                    pil_img = _auto_orient(pil_img)  # Fix rotation BEFORE resize
                    pil_img = _resize_if_needed(pil_img, max_size)
                    pil_img.save(preview_path, "JPEG", quality=85)
                    return preview_path
            except rawpy.LibRawNoThumbnailError:
                pass

            # Fall back to demosaic (slower but always works)
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # Faster, half resolution
                no_auto_bright=True,
            )
            pil_img = Image.fromarray(rgb)
            # Note: demosaic output is already in correct orientation
            pil_img = _resize_if_needed(pil_img, max_size)
            pil_img.save(preview_path, "JPEG", quality=85)
            return preview_path

    except Exception:
        return None


def _resize_if_needed(img: Image.Image, max_size: int) -> Image.Image:
    """Resize image if larger than max_size, preserving aspect ratio."""
    w, h = img.size
    if w <= max_size and h <= max_size:
        return img

    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _auto_orient(img: Image.Image) -> Image.Image:
    """Auto-rotate image based on EXIF orientation tag.

    Applied during preview extraction so all saved previews
    have correct orientation for subsequent analysis.
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
            return img
    except (AttributeError, KeyError, IndexError):
        return img


def get_preview_dir(index_dir: Path) -> Path:
    """Get or create the preview cache directory."""
    preview_dir = index_dir / ".qrate_previews"
    preview_dir.mkdir(exist_ok=True)
    return preview_dir
