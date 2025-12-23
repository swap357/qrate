"""Tests for qrate.export module."""

from pathlib import Path

from qrate.export import _generate_xmp, export_copy, export_list, export_xmp


class TestExportList:
    def test_writes_paths(self, tmp_path: Path):
        paths = ["/path/to/a.nef", "/path/to/b.nef"]
        out = tmp_path / "export.txt"
        export_list(paths, out)

        content = out.read_text()
        assert "/path/to/a.nef" in content
        assert "/path/to/b.nef" in content

    def test_includes_header(self, tmp_path: Path):
        paths = ["/path/to/a.nef"]
        out = tmp_path / "export.txt"
        export_list(paths, out)

        content = out.read_text()
        assert "# qrate export" in content
        assert "# count: 1" in content

    def test_includes_source_dir(self, tmp_path: Path):
        paths = ["/path/to/a.nef"]
        out = tmp_path / "export.txt"
        export_list(paths, out, source_dir=Path("/source"))

        content = out.read_text()
        assert "# source: /source" in content

    def test_includes_metadata(self, tmp_path: Path):
        paths = ["/path/to/a.nef"]
        out = tmp_path / "export.txt"
        export_list(paths, out, metadata={"filter": "sharpness > 100"})

        content = out.read_text()
        assert "# filter: sharpness > 100" in content


class TestExportCopy:
    def test_copies_files(self, tmp_path: Path):
        # Create source file
        src = tmp_path / "source"
        src.mkdir()
        (src / "a.nef").write_bytes(b"test data")

        # Copy
        out = tmp_path / "dest"
        count = export_copy([str(src / "a.nef")], out)

        assert count == 1
        assert (out / "a.nef").exists()
        assert (out / "a.nef").read_bytes() == b"test data"

    def test_handles_missing_source(self, tmp_path: Path):
        out = tmp_path / "dest"
        count = export_copy(["/nonexistent/file.nef"], out)
        assert count == 0

    def test_handles_name_collision(self, tmp_path: Path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "a.nef").write_bytes(b"data1")

        src2 = tmp_path / "source2"
        src2.mkdir()
        (src2 / "a.nef").write_bytes(b"data2")

        out = tmp_path / "dest"
        count = export_copy([str(src / "a.nef"), str(src2 / "a.nef")], out)

        assert count == 2
        assert (out / "a.nef").exists()
        assert (out / "a_1.nef").exists()


class TestExportXmp:
    def test_creates_sidecar(self, tmp_path: Path):
        nef = tmp_path / "photo.nef"
        nef.write_bytes(b"raw data")

        count = export_xmp([str(nef)], rating=5)

        assert count == 1
        xmp = tmp_path / "photo.nef.xmp"
        assert xmp.exists()
        content = xmp.read_text()
        assert 'xmp:Rating="5"' in content

    def test_skips_existing_xmp(self, tmp_path: Path):
        nef = tmp_path / "photo.nef"
        nef.write_bytes(b"raw data")
        xmp = tmp_path / "photo.nef.xmp"
        xmp.write_text("existing")

        count = export_xmp([str(nef)], rating=5)
        assert count == 0
        assert xmp.read_text() == "existing"

    def test_handles_missing_file(self, tmp_path: Path):
        count = export_xmp(["/nonexistent/file.nef"])
        assert count == 0

    def test_with_label(self, tmp_path: Path):
        nef = tmp_path / "photo.nef"
        nef.write_bytes(b"raw data")

        export_xmp([str(nef)], rating=4, label="Red")

        content = (tmp_path / "photo.nef.xmp").read_text()
        assert 'xmp:Label="Red"' in content


class TestGenerateXmp:
    def test_basic_xmp(self):
        content = _generate_xmp(rating=3)
        assert '<?xml version="1.0"' in content
        assert 'xmp:Rating="3"' in content

    def test_with_label(self):
        content = _generate_xmp(rating=5, label="Green")
        assert 'xmp:Label="Green"' in content

    def test_with_keywords(self):
        content = _generate_xmp(rating=4, keywords=["nature", "landscape"])
        assert "<rdf:li>nature</rdf:li>" in content
        assert "<rdf:li>landscape</rdf:li>" in content


class TestSelectForExport:
    def test_selects_images(self, tmp_path: Path):
        from qrate.db import ImageRecord, get_db
        from qrate.export import select_for_export

        db = get_db(tmp_path)
        db.upsert_image(ImageRecord(path="/a.nef"))
        db.upsert_image(ImageRecord(path="/b.nef"))

        paths = select_for_export(db)
        assert len(paths) == 2

    def test_limits_count(self, tmp_path: Path):
        from qrate.db import ImageRecord, get_db
        from qrate.export import select_for_export

        db = get_db(tmp_path)
        for i in range(10):
            db.upsert_image(ImageRecord(path=f"/{i}.nef"))

        paths = select_for_export(db, n=3)
        assert len(paths) == 3

    def test_with_sharpness_filter(self, tmp_path: Path):
        from qrate.db import ImageRecord, QualityScores, get_db
        from qrate.export import select_for_export

        db = get_db(tmp_path)
        db.upsert_image(ImageRecord(path="/sharp.nef"))
        db.upsert_image(ImageRecord(path="/blurry.nef"))
        db.upsert_quality(QualityScores(path="/sharp.nef", sharpness=1000))
        db.upsert_quality(QualityScores(path="/blurry.nef", sharpness=50))

        paths = select_for_export(db, min_sharpness=500)
        assert len(paths) == 1
        assert "/sharp.nef" in paths


class TestExportGallery:
    def test_export_gallery_no_images(self, tmp_path: Path):
        from qrate.db import get_db
        from qrate.export import export_gallery

        db = get_db(tmp_path)
        out_dir = tmp_path / "gallery"
        source_dir = tmp_path

        count = export_gallery(db, out_dir, source_dir, n=10)
        assert count == 0
        assert (out_dir / "scores.txt").exists()
        content = (out_dir / "scores.txt").read_text()
        assert "# count: 0" in content

    def test_export_gallery_with_images(self, tmp_path: Path):
        from PIL import Image

        from qrate.db import ImageRecord, get_db
        from qrate.export import export_gallery

        # Setup: create preview directory and images
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        preview_dir = source_dir / ".qrate_previews"
        preview_dir.mkdir(parents=True)

        # Create test images
        img1_path = source_dir / "photo1.nef"
        img1_path.write_bytes(b"raw1")
        preview1 = preview_dir / "photo1_preview.jpg"
        Image.new("RGB", (100, 100), color=(255, 0, 0)).save(preview1)

        img2_path = source_dir / "photo2.nef"
        img2_path.write_bytes(b"raw2")
        preview2 = preview_dir / "photo2_preview.jpg"
        Image.new("RGB", (100, 100), color=(0, 255, 0)).save(preview2)

        # Setup database
        db = get_db(tmp_path)
        db.upsert_image(ImageRecord(path=str(img1_path), preview_path=str(preview1)))
        db.upsert_image(ImageRecord(path=str(img2_path), preview_path=str(preview2)))

        # Export gallery
        out_dir = tmp_path / "gallery"
        count = export_gallery(db, out_dir, source_dir, n=10)

        assert count == 2
        assert (out_dir / "raw").exists()
        assert (out_dir / "jpg").exists()
        assert (out_dir / "scores.txt").exists()
        assert len(list((out_dir / "raw").glob("*.nef"))) == 2
        assert len(list((out_dir / "jpg").glob("*.jpg"))) == 2
