"""qrate: Select and export RAW file paths."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

__version__ = "0.1.0"

DEFAULT_EXTENSION = ".NEF"
DEFAULT_N = 200


def find_raw_files(directory: Path, extension: str = DEFAULT_EXTENSION) -> list[Path]:
    """Find all files with given extension in directory (recursive)."""
    return list(directory.rglob(f"*{extension}"))


def select_newest(files: list[Path], n: int) -> list[Path]:
    """Select up to n newest files by mtime."""
    sorted_files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return sorted_files[:n]


def write_export(
    out_path: Path,
    selected: list[Path],
    input_dir: Path,
    requested: int,
) -> None:
    """Write export file with header and paths."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# qrate v0",
        f"# input: {input_dir}",
        "# rule: newest_by_mtime",
        f"# requested: {requested}",
        f"# selected: {len(selected)}",
        f"# generated: {timestamp}",
        "",
    ]
    lines.extend(str(p.resolve()) for p in selected)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qrate",
        description="Select RAW files and export paths to text file.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory to scan")
    parser.add_argument("--out", type=Path, required=True, help="Output text file")
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N, help=f"Number of files (default: {DEFAULT_N})"
    )
    parser.add_argument(
        "--ext", default=DEFAULT_EXTENSION, help=f"File extension (default: {DEFAULT_EXTENSION})"
    )

    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        return 1

    files = find_raw_files(args.input_dir, args.ext)
    selected = select_newest(files, args.n)
    write_export(args.out, selected, args.input_dir, args.n)

    print(f"Wrote {len(selected)} paths to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
