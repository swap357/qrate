"""Parallel processing utilities for qrate.

Uses ProcessPoolExecutor for CPU-bound tasks like RAW demosaicing.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from qrate.db import ImageRecord, QualityScores


@dataclass
class IndexResult:
    """Result of indexing a single file."""

    path: str
    success: bool
    record: ImageRecord | None = None
    quality: QualityScores | None = None
    error: str | None = None


def _process_single_file(
    path_str: str,
    preview_dir_str: str,
) -> IndexResult:
    """Process a single file for indexing (runs in worker process).

    This function is designed to run in a separate process, so it
    imports everything it needs locally to avoid pickling issues.

    Args:
        path_str: Absolute path to RAW file.
        preview_dir_str: Path to preview cache directory.

    Returns:
        IndexResult with record and quality scores.
    """
    # Import inside function to avoid pickling module-level state
    from datetime import UTC

    from qrate.analyze import (
        compute_exposure_score,
        compute_file_hash,
        compute_perceptual_hash,
        compute_sharpness,
        estimate_noise,
    )
    from qrate.ingest import extract_exif, extract_preview

    path = Path(path_str)
    preview_dir = Path(preview_dir_str)

    try:
        stat = path.stat()

        # Extract EXIF
        exif = extract_exif(path)

        # Extract preview (CPU-intensive)
        preview_path = extract_preview(path, preview_dir)

        # Compute file hash
        file_hash = compute_file_hash(path)

        # Compute quality scores from preview
        phash = None
        sharpness = None
        exposure = None
        noise = None
        if preview_path and preview_path.exists():
            phash = compute_perceptual_hash(preview_path)
            sharpness = compute_sharpness(preview_path)
            exposure = compute_exposure_score(preview_path)
            noise = estimate_noise(preview_path, exif.iso)

        record = ImageRecord(
            path=path_str,
            hash_blake3=file_hash,
            hash_perceptual=phash,
            exif_timestamp=exif.timestamp,
            exif_iso=exif.iso,
            exif_shutter=exif.shutter,
            exif_aperture=exif.aperture,
            exif_focal_length=exif.focal_length,
            preview_path=str(preview_path) if preview_path else None,
            file_mtime=stat.st_mtime,
            file_size=stat.st_size,
            indexed_at=datetime.now(UTC),
        )

        quality = None
        if sharpness is not None:
            quality = QualityScores(
                path=path_str,
                sharpness=sharpness,
                exposure_score=exposure,
                noise_estimate=noise,
            )

        return IndexResult(
            path=path_str,
            success=True,
            record=record,
            quality=quality,
        )

    except Exception as e:
        return IndexResult(
            path=path_str,
            success=False,
            error=str(e),
        )


def process_files_parallel(
    files: list[Path],
    preview_dir: Path,
    workers: int | None = None,
) -> Iterator[IndexResult]:
    """Process multiple files in parallel.

    Args:
        files: List of file paths to process.
        preview_dir: Directory for preview cache.
        workers: Number of worker processes (default: CPU count).

    Yields:
        IndexResult for each processed file.
    """
    if workers is None:
        workers = os.cpu_count() or 4

    # Limit workers to reasonable bounds
    workers = max(1, min(workers, 16, len(files)))

    preview_dir_str = str(preview_dir)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                _process_single_file,
                str(path.resolve()),
                preview_dir_str,
            ): path
            for path in files
        }

        # Yield results as they complete
        for future in as_completed(futures):
            yield future.result()


def get_default_workers() -> int:
    """Get default number of workers based on CPU count."""
    cpu_count = os.cpu_count() or 4
    # Use N-1 CPUs to leave headroom, minimum 1
    return max(1, cpu_count - 1)
