"""Tests for qrate.group module."""

from pathlib import Path

from qrate.db import ImageRecord, get_db
from qrate.group import find_near_duplicates


class TestFindNearDuplicates:
    def test_empty_db(self, tmp_path: Path):
        db = get_db(tmp_path)
        result = find_near_duplicates(db)
        assert result == []

    def test_no_duplicates(self, tmp_path: Path):
        db = get_db(tmp_path)
        db.upsert_image(ImageRecord(path="/a.nef", hash_perceptual="0000000000000000"))
        db.upsert_image(ImageRecord(path="/b.nef", hash_perceptual="ffffffffffffffff"))
        result = find_near_duplicates(db, threshold=8)
        assert len(result) == 0

    def test_finds_duplicates(self, tmp_path: Path):
        db = get_db(tmp_path)
        db.upsert_image(ImageRecord(path="/a.nef", hash_perceptual="0000000000000000"))
        db.upsert_image(ImageRecord(path="/b.nef", hash_perceptual="0000000000000000"))
        result = find_near_duplicates(db, threshold=8)
        assert len(result) == 1

    def test_missing_hash_skipped(self, tmp_path: Path):
        db = get_db(tmp_path)
        db.upsert_image(ImageRecord(path="/a.nef", hash_perceptual=None))
        db.upsert_image(ImageRecord(path="/b.nef", hash_perceptual="0000000000000000"))
        result = find_near_duplicates(db, threshold=8)
        assert len(result) == 0
