"""Analyze module: image quality scoring and hashing."""

from __future__ import annotations

from pathlib import Path

import blake3
import imagehash
import numpy as np
from numpy.typing import NDArray
from PIL import Image

# Try to import opencv, fall back to scipy
try:
    import cv2  # type: ignore[import-not-found]

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def compute_sharpness(image: Image.Image | Path | str) -> float:
    """Compute sharpness score using variance of Laplacian.

    Higher values = sharper image.

    Args:
        image: PIL Image, or path to image file.

    Returns:
        Sharpness score (variance of Laplacian).
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    # Convert to grayscale numpy array
    gray = np.array(image.convert("L"), dtype=np.float64)

    if HAS_CV2:
        # OpenCV Laplacian (faster, more accurate)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    else:
        # Pure numpy/scipy fallback using convolution
        return _laplacian_variance_numpy(gray)


def _laplacian_variance_numpy(gray: NDArray[np.float64]) -> float:
    """Compute Laplacian variance using numpy convolution."""
    # Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    # Simple convolution (not as fast as cv2 but works)
    from scipy import ndimage  # type: ignore[import-untyped]

    laplacian = ndimage.convolve(gray, kernel)
    return float(laplacian.var())


def compute_exposure_score(image: Image.Image | Path | str) -> float:
    """Compute exposure quality score based on histogram analysis.

    Score is 1.0 for well-exposed, lower for over/underexposed.

    Args:
        image: PIL Image, or path to image file.

    Returns:
        Exposure score from 0.0 (terrible) to 1.0 (good).
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    # Convert to grayscale
    gray = image.convert("L")
    hist = np.array(gray.histogram(), dtype=np.float64)

    # Normalize histogram
    hist = hist / hist.sum()

    # Compute penalties for clipping
    # Clipping = too many pixels at 0 (underexposed) or 255 (overexposed)
    underexposed = hist[:16].sum()  # Bottom ~6% of range
    overexposed = hist[-16:].sum()  # Top ~6% of range

    # Penalty for extreme clipping (more than 5% of pixels)
    clip_penalty = max(0, underexposed - 0.05) + max(0, overexposed - 0.05)

    # Compute histogram spread (well-exposed images use more of the range)
    nonzero_bins = np.count_nonzero(hist)
    spread_score = nonzero_bins / 256.0

    # Compute center of mass (should be around 128 for balanced exposure)
    center = np.average(np.arange(256), weights=hist)
    center_deviation = abs(center - 128) / 128.0

    # Combine scores
    score = spread_score * (1 - clip_penalty) * (1 - 0.3 * center_deviation)
    return max(0.0, min(1.0, score))


def estimate_noise(image: Image.Image | Path | str, iso: int | None = None) -> float:
    """Estimate noise level in image.

    Args:
        image: PIL Image, or path to image file.
        iso: Optional ISO value from EXIF (higher = more noise expected).

    Returns:
        Noise estimate from 0.0 (clean) to 1.0 (very noisy).
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    # Simple noise estimation using local variance
    gray = np.array(image.convert("L"), dtype=np.float64)

    # Use median absolute deviation of Laplacian as noise estimate
    if HAS_CV2:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    else:
        from scipy import ndimage  # type: ignore[import-untyped]

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        laplacian = ndimage.convolve(gray, kernel)

    # Median absolute deviation (robust noise estimator)
    sigma = np.median(np.abs(laplacian)) / 0.6745

    # Normalize to 0-1 range (empirical scaling)
    noise_score = min(1.0, sigma / 50.0)

    # Adjust for ISO if available (high ISO = expect more noise)
    if iso and iso > 800:
        # Don't penalize as much if high ISO noise is expected
        expected_noise = min(1.0, (iso - 800) / 10000)
        # Score is how much noise exceeds expectation
        noise_score = max(0.0, noise_score - expected_noise * 0.5)

    return noise_score


def compute_file_hash(path: Path) -> str:
    """Compute blake3 hash of file contents.

    Args:
        path: Path to file.

    Returns:
        Hex string of blake3 hash.
    """
    hasher = blake3.blake3()
    with open(path, "rb") as f:
        # Read in chunks for large files
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_perceptual_hash(image: Image.Image | Path | str) -> str:
    """Compute perceptual hash for near-duplicate detection.

    Uses pHash (perceptual hash) which is robust to resizing and
    minor edits.

    Args:
        image: PIL Image, or path to image file.

    Returns:
        Hex string of 64-bit perceptual hash.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    # imagehash.phash returns ImageHash object
    phash = imagehash.phash(image)
    return str(phash)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two perceptual hashes.

    Args:
        hash1: First hash (hex string).
        hash2: Second hash (hex string).

    Returns:
        Number of differing bits (0 = identical, higher = more different).
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def are_near_duplicates(hash1: str, hash2: str, threshold: int = 8) -> bool:
    """Check if two images are near-duplicates based on perceptual hash.

    Args:
        hash1: First perceptual hash.
        hash2: Second perceptual hash.
        threshold: Maximum Hamming distance to consider duplicates.
                  Default 8 is a good balance for photo detection.

    Returns:
        True if images are likely near-duplicates.
    """
    return hamming_distance(hash1, hash2) <= threshold
