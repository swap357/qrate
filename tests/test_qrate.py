"""Tests for qrate."""

import time
from pathlib import Path

from qrate import RAW_EXTENSIONS, find_raw_files, main, select_newest, write_export


def test_find_raw_files(tmp_path: Path):
    """Find RAW files recursively."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (tmp_path / "a.NEF").touch()
    (subdir / "b.CR2").touch()
    (tmp_path / "c.ARW").touch()
    (tmp_path / "d.jpg").touch()

    files = find_raw_files(tmp_path)
    assert len(files) == 3
    assert all(f.suffix.lower() in RAW_EXTENSIONS for f in files)


def test_find_raw_files_case_insensitive(tmp_path: Path):
    """Find RAW files regardless of case."""
    (tmp_path / "a.nef").touch()
    (tmp_path / "b.NEF").touch()
    (tmp_path / "c.Nef").touch()

    files = find_raw_files(tmp_path)
    assert len(files) == 3


def test_find_raw_files_custom_extensions(tmp_path: Path):
    """Find files with custom extensions."""
    (tmp_path / "a.NEF").touch()
    (tmp_path / "b.CR2").touch()

    files = find_raw_files(tmp_path, frozenset({".nef"}))
    assert len(files) == 1
    assert files[0].suffix == ".NEF"


def test_select_newest(tmp_path: Path):
    """Select newest by mtime."""
    old = tmp_path / "old.NEF"
    new = tmp_path / "new.NEF"
    old.touch()
    time.sleep(0.01)
    new.touch()

    selected = select_newest([old, new], n=1)
    assert len(selected) == 1
    assert selected[0] == new


def test_select_newest_limits(tmp_path: Path):
    """Respect n limit."""
    files = [tmp_path / f"{i}.NEF" for i in range(10)]
    for f in files:
        f.touch()

    selected = select_newest(files, n=3)
    assert len(selected) == 3


def test_write_export(tmp_path: Path):
    """Write export file with correct format."""
    nef = tmp_path / "test.NEF"
    nef.touch()
    out = tmp_path / "export.txt"

    write_export(out, [nef], tmp_path, requested=100)

    content = out.read_text()
    assert content.startswith("# qrate v0\n")
    assert f"# input: {tmp_path}\n" in content
    assert "# rule: newest_by_mtime\n" in content
    assert "# requested: 100\n" in content
    assert "# selected: 1\n" in content
    assert "# generated: " in content
    assert str(nef.resolve()) in content


def test_main_integration(tmp_path: Path):
    """Full integration test."""
    (tmp_path / "input").mkdir()
    for i in range(5):
        (tmp_path / "input" / f"IMG_{i}.NEF").touch()
        time.sleep(0.01)

    out = tmp_path / "export.txt"
    result = main(
        [
            "select",
            str(tmp_path / "input"),
            "--out",
            str(out),
            "--n",
            "3",
            "--ext",
            ".NEF",
        ]
    )

    assert result == 0
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    header_lines = [ln for ln in lines if ln.startswith("#")]
    path_lines = [ln for ln in lines if not ln.startswith("#") and ln]
    assert len(header_lines) == 6
    assert len(path_lines) == 3


def test_main_empty_dir(tmp_path: Path):
    """Handle empty directory."""
    (tmp_path / "empty").mkdir()
    out = tmp_path / "export.txt"

    result = main(
        ["select", str(tmp_path / "empty"), "--out", str(out), "--ext", ".NEF"]
    )

    assert result == 0
    content = out.read_text()
    assert "# selected: 0\n" in content


def test_main_invalid_dir(tmp_path: Path):
    """Error on invalid directory."""
    result = main(["select", str(tmp_path / "nonexistent"), "--out", "out.txt"])
    assert result == 1


def test_index_command(tmp_path: Path):
    """Test index command."""
    (tmp_path / "a.NEF").touch()
    (tmp_path / "b.CR2").touch()

    result = main(["index", str(tmp_path)])
    assert result == 0

    # Check database was created
    db_path = tmp_path / ".qrate.db"
    assert db_path.exists()


def test_status_command(tmp_path: Path):
    """Test status command."""
    (tmp_path / "a.NEF").touch()
    main(["index", str(tmp_path)])

    result = main(["status", str(tmp_path)])
    assert result == 0


def test_index_incremental(tmp_path: Path):
    """Test that index is incremental."""
    (tmp_path / "a.NEF").touch()
    main(["index", str(tmp_path)])

    # Second index should skip unchanged files
    result = main(["index", str(tmp_path)])
    assert result == 0


def test_group_bursts_command(tmp_path: Path):
    """Test group-bursts command."""
    (tmp_path / "a.NEF").touch()
    main(["index", str(tmp_path)])

    result = main(["group-bursts", str(tmp_path)])
    assert result == 0


def test_export_list_command(tmp_path: Path):
    """Test export command with list format."""
    (tmp_path / "a.NEF").touch()
    main(["index", str(tmp_path)])

    out = tmp_path / "export.txt"
    result = main(["export", str(tmp_path), "--out", str(out), "--format", "list"])
    assert result == 0
    assert out.exists()


def test_export_copy_command(tmp_path: Path):
    """Test export command with copy format."""
    (tmp_path / "a.NEF").write_bytes(b"fake raw data")
    main(["index", str(tmp_path)])

    out_dir = tmp_path / "exported"
    result = main(["export", str(tmp_path), "--out", str(out_dir), "--format", "copy"])
    assert result == 0
    assert out_dir.exists()
    assert len(list(out_dir.iterdir())) == 1


def test_export_xmp_command(tmp_path: Path):
    """Test export command with xmp format."""
    nef = tmp_path / "a.NEF"
    nef.write_bytes(b"fake raw data")
    main(["index", str(tmp_path)])

    # XMP format writes sidecars next to originals, not to --out
    result = main(
        [
            "export",
            str(tmp_path),
            "--out",
            str(tmp_path),
            "--format",
            "xmp",
            "--rating",
            "4",
        ]
    )
    assert result == 0
    assert (tmp_path / "a.NEF.xmp").exists()


def test_score_command(tmp_path: Path):
    """Test score command."""
    # Create a fake preview
    preview_dir = tmp_path / ".qrate_previews"
    preview_dir.mkdir()

    # Create minimal JPEG
    from PIL import Image

    img = Image.new("RGB", (100, 100), (128, 128, 128))
    img.save(preview_dir / "a_preview.jpg", "JPEG")

    (tmp_path / "a.NEF").touch()
    main(["index", str(tmp_path)])

    result = main(["score", str(tmp_path), "--top", "5"])
    assert result == 0


def test_score_command_verbose(tmp_path: Path):
    """Test score command with verbose output."""
    preview_dir = tmp_path / ".qrate_previews"
    preview_dir.mkdir()

    from PIL import Image

    img = Image.new("RGB", (100, 100), (128, 128, 128))
    img.save(preview_dir / "a_preview.jpg", "JPEG")

    (tmp_path / "a.NEF").touch()
    main(["index", str(tmp_path)])

    result = main(["score", str(tmp_path), "--top", "5", "--verbose"])
    assert result == 0


def test_score_command_no_images(tmp_path: Path):
    """Test score command with no images."""
    result = main(["score", str(tmp_path)])
    assert result == 1  # Should fail - no images indexed


def test_group_bursts_no_images(tmp_path: Path):
    """Test group-bursts command with no indexed images."""
    result = main(["group-bursts", str(tmp_path)])
    assert result == 1


def test_export_no_images(tmp_path: Path):
    """Test export command with no indexed images."""
    result = main(["export", str(tmp_path), "--out", str(tmp_path / "out.txt")])
    assert result == 1


def test_status_invalid_dir(tmp_path: Path):
    """Test status on invalid directory."""
    result = main(["status", str(tmp_path / "nonexistent")])
    assert result == 1


def test_index_invalid_dir(tmp_path: Path):
    """Test index on invalid directory."""
    result = main(["index", str(tmp_path / "nonexistent")])
    assert result == 1
