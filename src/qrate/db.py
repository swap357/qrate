"""SQLite database for qrate image index."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Default database location
DEFAULT_DB_NAME = ".qrate.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY,
    hash_blake3 TEXT,
    hash_perceptual TEXT,
    exif_timestamp TEXT,
    exif_iso INTEGER,
    exif_shutter REAL,
    exif_aperture REAL,
    exif_focal_length REAL,
    preview_path TEXT,
    file_mtime REAL,
    file_size INTEGER,
    indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS quality_scores (
    path TEXT PRIMARY KEY REFERENCES images(path) ON DELETE CASCADE,
    sharpness REAL,
    exposure_score REAL,
    noise_estimate REAL
);

CREATE TABLE IF NOT EXISTS burst_groups (
    group_id INTEGER,
    path TEXT REFERENCES images(path) ON DELETE CASCADE,
    is_best INTEGER DEFAULT 0,
    PRIMARY KEY (group_id, path)
);

CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(exif_timestamp);
CREATE INDEX IF NOT EXISTS idx_images_hash_blake3 ON images(hash_blake3);
CREATE INDEX IF NOT EXISTS idx_images_hash_perceptual ON images(hash_perceptual);
CREATE INDEX IF NOT EXISTS idx_burst_groups_group_id ON burst_groups(group_id);
"""


@dataclass
class ImageRecord:
    """A single indexed image."""

    path: str
    hash_blake3: str | None = None
    hash_perceptual: str | None = None
    exif_timestamp: datetime | None = None
    exif_iso: int | None = None
    exif_shutter: float | None = None
    exif_aperture: float | None = None
    exif_focal_length: float | None = None
    preview_path: str | None = None
    file_mtime: float | None = None
    file_size: int | None = None
    indexed_at: datetime | None = None


@dataclass
class QualityScores:
    """Quality metrics for an image."""

    path: str
    sharpness: float | None = None
    exposure_score: float | None = None
    noise_estimate: float | None = None


@dataclass
class BurstMember:
    """A member of a burst group."""

    group_id: int
    path: str
    is_best: bool = False


class Database:
    """SQLite database wrapper for qrate index."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection, creating schema if needed."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Initialize database schema."""
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def upsert_image(self, record: ImageRecord) -> None:
        """Insert or update an image record."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO images (
                    path, hash_blake3, hash_perceptual, exif_timestamp,
                    exif_iso, exif_shutter, exif_aperture, exif_focal_length,
                    preview_path, file_mtime, file_size, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    hash_blake3 = excluded.hash_blake3,
                    hash_perceptual = excluded.hash_perceptual,
                    exif_timestamp = excluded.exif_timestamp,
                    exif_iso = excluded.exif_iso,
                    exif_shutter = excluded.exif_shutter,
                    exif_aperture = excluded.exif_aperture,
                    exif_focal_length = excluded.exif_focal_length,
                    preview_path = excluded.preview_path,
                    file_mtime = excluded.file_mtime,
                    file_size = excluded.file_size,
                    indexed_at = excluded.indexed_at
                """,
                (
                    record.path,
                    record.hash_blake3,
                    record.hash_perceptual,
                    record.exif_timestamp.isoformat()
                    if record.exif_timestamp
                    else None,
                    record.exif_iso,
                    record.exif_shutter,
                    record.exif_aperture,
                    record.exif_focal_length,
                    record.preview_path,
                    record.file_mtime,
                    record.file_size,
                    record.indexed_at.isoformat() if record.indexed_at else None,
                ),
            )
            conn.commit()

    def get_image(self, path: str) -> ImageRecord | None:
        """Get an image record by path."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE path = ?", (path,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_image(row)

    def get_all_images(self) -> list[ImageRecord]:
        """Get all indexed images."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM images ORDER BY exif_timestamp DESC"
            ).fetchall()
            return [self._row_to_image(row) for row in rows]

    def get_images_needing_index(
        self, paths_with_mtime: list[tuple[str, float]]
    ) -> list[str]:
        """Return paths that are new or have changed mtime."""
        with self.connection() as conn:
            result = []
            for path, mtime in paths_with_mtime:
                row = conn.execute(
                    "SELECT file_mtime FROM images WHERE path = ?", (path,)
                ).fetchone()
                if row is None or row["file_mtime"] != mtime:
                    result.append(path)
            return result

    def upsert_quality(self, scores: QualityScores) -> None:
        """Insert or update quality scores."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO quality_scores (path, sharpness, exposure_score, noise_estimate)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    sharpness = excluded.sharpness,
                    exposure_score = excluded.exposure_score,
                    noise_estimate = excluded.noise_estimate
                """,
                (
                    scores.path,
                    scores.sharpness,
                    scores.exposure_score,
                    scores.noise_estimate,
                ),
            )
            conn.commit()

    def get_quality(self, path: str) -> QualityScores | None:
        """Get quality scores for an image."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM quality_scores WHERE path = ?", (path,)
            ).fetchone()
            if row is None:
                return None
            return QualityScores(
                path=row["path"],
                sharpness=row["sharpness"],
                exposure_score=row["exposure_score"],
                noise_estimate=row["noise_estimate"],
            )

    def set_burst_group(
        self, group_id: int, paths: list[str], best_path: str | None = None
    ) -> None:
        """Set burst group membership."""
        with self.connection() as conn:
            # Clear existing membership for these paths
            placeholders = ",".join("?" * len(paths))
            conn.execute(
                f"DELETE FROM burst_groups WHERE path IN ({placeholders})", paths
            )
            # Insert new membership
            for path in paths:
                conn.execute(
                    "INSERT INTO burst_groups (group_id, path, is_best) VALUES (?, ?, ?)",
                    (group_id, path, 1 if path == best_path else 0),
                )
            conn.commit()

    def get_burst_groups(self) -> dict[int, list[BurstMember]]:
        """Get all burst groups."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT group_id, path, is_best FROM burst_groups ORDER BY group_id"
            ).fetchall()
            groups: dict[int, list[BurstMember]] = {}
            for row in rows:
                gid = row["group_id"]
                if gid not in groups:
                    groups[gid] = []
                groups[gid].append(
                    BurstMember(
                        group_id=gid, path=row["path"], is_best=bool(row["is_best"])
                    )
                )
            return groups

    def get_best_of_bursts(self) -> list[str]:
        """Get paths of best images from each burst group."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT path FROM burst_groups WHERE is_best = 1"
            ).fetchall()
            return [row["path"] for row in rows]

    def get_standalone_images(self) -> list[str]:
        """Get paths of images not in any burst group."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT path FROM images
                WHERE path NOT IN (SELECT path FROM burst_groups)
                """
            ).fetchall()
            return [row["path"] for row in rows]

    def count_images(self) -> int:
        """Count total indexed images."""
        with self.connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM images").fetchone()
            return row["cnt"] if row else 0

    def count_duplicates(self) -> int:
        """Count images with duplicate blake3 hashes."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM images
                WHERE hash_blake3 IN (
                    SELECT hash_blake3 FROM images
                    GROUP BY hash_blake3 HAVING COUNT(*) > 1
                )
                """
            ).fetchone()
            return row["cnt"] if row else 0

    def find_exact_duplicates(self) -> list[list[str]]:
        """Find groups of exact duplicates by blake3 hash."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT hash_blake3, GROUP_CONCAT(path) as paths
                FROM images
                WHERE hash_blake3 IS NOT NULL
                GROUP BY hash_blake3
                HAVING COUNT(*) > 1
                """
            ).fetchall()
            return [row["paths"].split(",") for row in rows]

    def delete_image(self, path: str) -> None:
        """Delete an image and its related records."""
        with self.connection() as conn:
            conn.execute("DELETE FROM images WHERE path = ?", (path,))
            conn.commit()

    def _row_to_image(self, row: sqlite3.Row) -> ImageRecord:
        """Convert a database row to ImageRecord."""
        return ImageRecord(
            path=row["path"],
            hash_blake3=row["hash_blake3"],
            hash_perceptual=row["hash_perceptual"],
            exif_timestamp=(
                datetime.fromisoformat(row["exif_timestamp"])
                if row["exif_timestamp"]
                else None
            ),
            exif_iso=row["exif_iso"],
            exif_shutter=row["exif_shutter"],
            exif_aperture=row["exif_aperture"],
            exif_focal_length=row["exif_focal_length"],
            preview_path=row["preview_path"],
            file_mtime=row["file_mtime"],
            file_size=row["file_size"],
            indexed_at=(
                datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None
            ),
        )


def get_db(directory: Path) -> Database:
    """Get database for a directory, creating if needed."""
    db_path = directory / DEFAULT_DB_NAME
    db = Database(db_path)
    db.init_schema()
    return db
