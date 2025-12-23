"""Tests for qrate.ingest module."""

from pathlib import Path

from PIL import Image

from qrate.ingest import ExifData, _auto_orient, _resize_if_needed, get_preview_dir


class TestExifData:
    def test_defaults(self):
        exif = ExifData()
        assert exif.timestamp is None
        assert exif.iso is None
        assert exif.shutter is None
        assert exif.aperture is None
        assert exif.focal_length is None
        assert exif.camera_make is None
        assert exif.camera_model is None

    def test_custom_values(self):
        from datetime import datetime

        ts = datetime.now()
        exif = ExifData(timestamp=ts, iso=100, shutter=0.01, aperture=2.8)
        assert exif.timestamp == ts
        assert exif.iso == 100
        assert exif.shutter == 0.01
        assert exif.aperture == 2.8


class TestResizeIfNeeded:
    def test_small_image_unchanged(self):
        img = Image.new("RGB", (500, 300))
        result = _resize_if_needed(img, 1024)
        assert result.size == (500, 300)

    def test_large_width_resized(self):
        img = Image.new("RGB", (2000, 1000))
        result = _resize_if_needed(img, 1024)
        assert result.size[0] == 1024
        assert result.size[1] == 512

    def test_large_height_resized(self):
        img = Image.new("RGB", (1000, 2000))
        result = _resize_if_needed(img, 1024)
        assert result.size[0] == 512
        assert result.size[1] == 1024

    def test_exact_size_unchanged(self):
        img = Image.new("RGB", (1024, 1024))
        result = _resize_if_needed(img, 1024)
        assert result.size == (1024, 1024)


class TestAutoOrient:
    def test_no_exif(self):
        img = Image.new("RGB", (100, 100))
        result = _auto_orient(img)
        assert result.size == (100, 100)

    def test_returns_image(self):
        img = Image.new("RGB", (100, 100))
        result = _auto_orient(img)
        assert isinstance(result, Image.Image)


class TestGetPreviewDir:
    def test_creates_directory(self, tmp_path: Path):
        preview_dir = get_preview_dir(tmp_path)
        assert preview_dir.exists()
        assert preview_dir.name == ".qrate_previews"

    def test_returns_existing(self, tmp_path: Path):
        preview_dir = get_preview_dir(tmp_path)
        preview_dir2 = get_preview_dir(tmp_path)
        assert preview_dir == preview_dir2


class TestExtractExif:
    def test_returns_exif_data(self):
        from qrate.ingest import ExifData, extract_exif

        # Should return ExifData even for non-existent/invalid files
        result = extract_exif(Path("/nonexistent/file.nef"))
        assert isinstance(result, ExifData)
