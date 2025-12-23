"""Tests for qrate."""

import time
from pathlib import Path

from qrate import find_raw_files, main, select_newest, write_export


def test_find_raw_files(tmp_path: Path):
    """Find .NEF files recursively."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (tmp_path / "a.NEF").touch()
    (subdir / "b.NEF").touch()
    (tmp_path / "c.jpg").touch()

    files = find_raw_files(tmp_path, ".NEF")
    assert len(files) == 2
    assert all(f.suffix == ".NEF" for f in files)


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
    result = main([str(tmp_path / "input"), "--out", str(out), "--n", "3"])

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

    result = main([str(tmp_path / "empty"), "--out", str(out)])

    assert result == 0
    content = out.read_text()
    assert "# selected: 0\n" in content


def test_main_invalid_dir(tmp_path: Path):
    """Error on invalid directory."""
    result = main([str(tmp_path / "nonexistent"), "--out", "out.txt"])
    assert result == 1
