"""Tests for qrate.db module."""

from datetime import UTC, datetime
from pathlib import Path

from qrate.db import ImageRecord, QualityScores, get_db


def test_database_init(tmp_path: Path):
    """Database initializes schema on creation."""
    db = get_db(tmp_path)
    assert db.db_path.exists()
    assert db.count_images() == 0


def test_upsert_and_get_image(tmp_path: Path):
    """Insert and retrieve an image record."""
    db = get_db(tmp_path)

    record = ImageRecord(
        path="/test/image.nef",
        hash_blake3="abc123",
        exif_iso=400,
        file_mtime=1234567890.0,
        file_size=1024,
        indexed_at=datetime.now(UTC),
    )
    db.upsert_image(record)

    retrieved = db.get_image("/test/image.nef")
    assert retrieved is not None
    assert retrieved.hash_blake3 == "abc123"
    assert retrieved.exif_iso == 400


def test_upsert_updates_existing(tmp_path: Path):
    """Upsert updates existing record."""
    db = get_db(tmp_path)

    record1 = ImageRecord(path="/test/image.nef", exif_iso=400)
    db.upsert_image(record1)

    record2 = ImageRecord(path="/test/image.nef", exif_iso=800)
    db.upsert_image(record2)

    retrieved = db.get_image("/test/image.nef")
    assert retrieved is not None
    assert retrieved.exif_iso == 800
    assert db.count_images() == 1


def test_get_all_images(tmp_path: Path):
    """Get all images from database."""
    db = get_db(tmp_path)

    for i in range(3):
        db.upsert_image(ImageRecord(path=f"/test/{i}.nef"))

    images = db.get_all_images()
    assert len(images) == 3


def test_images_needing_index(tmp_path: Path):
    """Identify files needing reindex."""
    db = get_db(tmp_path)

    db.upsert_image(ImageRecord(path="/a.nef", file_mtime=100.0))
    db.upsert_image(ImageRecord(path="/b.nef", file_mtime=200.0))

    # a.nef unchanged, b.nef changed, c.nef new
    needs = db.get_images_needing_index(
        [
            ("/a.nef", 100.0),
            ("/b.nef", 300.0),
            ("/c.nef", 400.0),
        ]
    )

    assert "/a.nef" not in needs
    assert "/b.nef" in needs
    assert "/c.nef" in needs


def test_quality_scores(tmp_path: Path):
    """Insert and retrieve quality scores."""
    db = get_db(tmp_path)
    db.upsert_image(ImageRecord(path="/test.nef"))

    scores = QualityScores(path="/test.nef", sharpness=123.5, exposure_score=0.8)
    db.upsert_quality(scores)

    retrieved = db.get_quality("/test.nef")
    assert retrieved is not None
    assert retrieved.sharpness == 123.5
    assert retrieved.exposure_score == 0.8


def test_burst_groups(tmp_path: Path):
    """Set and retrieve burst groups."""
    db = get_db(tmp_path)

    for path in ["/a.nef", "/b.nef", "/c.nef"]:
        db.upsert_image(ImageRecord(path=path))

    db.set_burst_group(1, ["/a.nef", "/b.nef"], best_path="/a.nef")
    db.set_burst_group(2, ["/c.nef"], best_path="/c.nef")

    groups = db.get_burst_groups()
    assert len(groups) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 1

    best = db.get_best_of_bursts()
    assert "/a.nef" in best
    assert "/c.nef" in best
    assert "/b.nef" not in best


def test_standalone_images(tmp_path: Path):
    """Get images not in any burst group."""
    db = get_db(tmp_path)

    for path in ["/a.nef", "/b.nef", "/c.nef"]:
        db.upsert_image(ImageRecord(path=path))

    db.set_burst_group(1, ["/a.nef", "/b.nef"])

    standalone = db.get_standalone_images()
    assert standalone == ["/c.nef"]


def test_find_exact_duplicates(tmp_path: Path):
    """Find images with same blake3 hash."""
    db = get_db(tmp_path)

    db.upsert_image(ImageRecord(path="/a.nef", hash_blake3="hash1"))
    db.upsert_image(ImageRecord(path="/b.nef", hash_blake3="hash1"))
    db.upsert_image(ImageRecord(path="/c.nef", hash_blake3="hash2"))

    dupes = db.find_exact_duplicates()
    assert len(dupes) == 1
    assert set(dupes[0]) == {"/a.nef", "/b.nef"}


def test_delete_image(tmp_path: Path):
    """Delete an image and its related records."""
    db = get_db(tmp_path)

    db.upsert_image(ImageRecord(path="/test.nef"))
    assert db.count_images() == 1

    db.delete_image("/test.nef")
    assert db.count_images() == 0
