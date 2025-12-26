"""Group module: deduplication and burst detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from qrate.analyze import are_near_duplicates
from qrate.db import Database, ImageRecord


@dataclass
class Burst:
    """A group of images taken in quick succession."""

    group_id: int
    images: list[ImageRecord]
    best_path: str | None = None


def find_exact_duplicates(db: Database) -> list[list[str]]:
    """Find groups of exact duplicates by file hash.

    Args:
        db: Database instance.

    Returns:
        List of groups, each group is a list of duplicate paths.
    """
    return db.find_exact_duplicates()


def find_near_duplicates(db: Database, threshold: int = 8) -> list[list[str]]:
    """Find groups of near-duplicates by perceptual hash using LSH.

    Uses band-based locality-sensitive hashing to reduce O(n²) to ~O(n).

    Args:
        db: Database instance.
        threshold: Hamming distance threshold (lower = stricter).

    Returns:
        List of groups, each group is a list of near-duplicate paths.
    """
    from collections import defaultdict

    images = db.get_all_images()
    with_hash = [
        (img.path, img.hash_perceptual) for img in images if img.hash_perceptual
    ]

    if not with_hash:
        return []

    # LSH: Split 16-char hex hash into 4 bands of 4 chars each
    # Hashes sharing any band are candidates for comparison
    bands: dict[str, list[int]] = defaultdict(list)
    for idx, (_, phash) in enumerate(with_hash):
        for b in range(4):
            band_key = f"{b}:{phash[b * 4 : (b + 1) * 4]}"
            bands[band_key].append(idx)

    # Union-find
    parent: dict[str, str] = {path: path for path, _ in with_hash}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare only within same bands (candidate pairs)
    compared: set[tuple[int, int]] = set()
    for indices in bands.values():
        if len(indices) < 2:
            continue
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i + 1 :]:
                pair = (min(idx1, idx2), max(idx1, idx2))
                if pair in compared:
                    continue
                compared.add(pair)
                path1, hash1 = with_hash[idx1]
                path2, hash2 = with_hash[idx2]
                if are_near_duplicates(hash1, hash2, threshold):
                    union(path1, path2)

    # Group by root
    groups: dict[str, list[str]] = defaultdict(list)
    for path, _ in with_hash:
        groups[find(path)].append(path)

    return [g for g in groups.values() if len(g) > 1]


def detect_bursts(
    db: Database,
    time_threshold: timedelta = timedelta(seconds=2),
    phash_threshold: int = 12,
) -> list[Burst]:
    """Detect burst sequences (rapid shots of similar scenes).

    A burst is defined as images:
    - Taken within time_threshold of each other
    - With similar perceptual hashes (optional, helps with timestamp errors)

    Args:
        db: Database instance.
        time_threshold: Maximum time between consecutive burst images.
        phash_threshold: Maximum Hamming distance for perceptual similarity.

    Returns:
        List of detected burst groups.
    """
    images = db.get_all_images()

    # Filter to images with timestamps
    with_time = [img for img in images if img.exif_timestamp]
    with_time.sort(key=lambda x: x.exif_timestamp)  # type: ignore[arg-type, return-value]

    if not with_time:
        return []

    bursts: list[Burst] = []
    current_burst: list[ImageRecord] = [with_time[0]]
    group_id = 0

    for img in with_time[1:]:
        prev = current_burst[-1]
        time_diff = img.exif_timestamp - prev.exif_timestamp  # type: ignore[operator]

        # Check time proximity
        in_time_window = time_diff <= time_threshold

        # Check perceptual similarity (if both have hashes)
        similar = True
        if prev.hash_perceptual and img.hash_perceptual:
            similar = are_near_duplicates(
                prev.hash_perceptual, img.hash_perceptual, phash_threshold
            )

        if in_time_window and similar:
            current_burst.append(img)
        else:
            # Save current burst if it has multiple images
            if len(current_burst) > 1:
                bursts.append(Burst(group_id=group_id, images=current_burst))
                group_id += 1
            current_burst = [img]

    # Don't forget the last burst
    if len(current_burst) > 1:
        bursts.append(Burst(group_id=group_id, images=current_burst))

    return bursts


def select_best_of_burst(burst: Burst, db: Database) -> str:
    """Select the best image from a burst based on quality scores.

    Selection criteria (in order of importance):
    1. Sharpness (higher is better)
    2. Exposure score (higher is better)
    3. Lower noise estimate

    Args:
        burst: Burst group to analyze.
        db: Database instance (for quality scores).

    Returns:
        Path of the best image in the burst.
    """
    if len(burst.images) == 1:
        return burst.images[0].path

    scores: list[tuple[str, float]] = []

    for img in burst.images:
        quality = db.get_quality(img.path)
        if quality:
            # Composite score: sharpness + exposure - noise
            score = (
                (quality.sharpness or 0)
                + (quality.exposure_score or 0) * 100
                - (quality.noise_estimate or 0) * 50
            )
            scores.append((img.path, score))
        else:
            # No quality data, use file size as proxy (larger = more data = potentially better)
            scores.append((img.path, float(img.file_size or 0)))

    # Return path with highest score
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def process_bursts(db: Database, time_threshold: float = 2.0) -> int:
    """Detect bursts and store results in database.

    Args:
        db: Database instance.
        time_threshold: Seconds between burst images.

    Returns:
        Number of bursts detected.
    """
    bursts = detect_bursts(db, timedelta(seconds=time_threshold))

    for burst in bursts:
        best = select_best_of_burst(burst, db)
        burst.best_path = best
        paths = [img.path for img in burst.images]
        db.set_burst_group(burst.group_id, paths, best)

    return len(bursts)
