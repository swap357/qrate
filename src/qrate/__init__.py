"""qrate: Image curator."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

__version__ = "0.1.0"

# Supported RAW formats (case-insensitive matching)
RAW_EXTENSIONS = frozenset({".nef", ".cr2", ".arw", ".dng"})
DEFAULT_N = 200


def find_raw_files(
    directory: Path,
    extensions: frozenset[str] | None = None,
) -> list[Path]:
    """Find all RAW files in directory (recursive).

    Args:
        directory: Root directory to scan.
        extensions: Set of extensions to match (lowercase, with dot).
                   Defaults to RAW_EXTENSIONS.

    Returns:
        List of paths to RAW files.
    """
    if extensions is None:
        extensions = RAW_EXTENSIONS

    result = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            result.append(path)
    return result


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


def _parse_extensions(ext_arg: str) -> frozenset[str]:
    """Parse comma-separated extensions into a frozenset."""
    exts = []
    for e in ext_arg.split(","):
        e = e.strip().lower()
        if not e.startswith("."):
            e = "." + e
        exts.append(e)
    return frozenset(exts)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qrate",
        description="Image curator.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # index command
    index_parser = subparsers.add_parser("index", help="Index a directory of RAW files")
    index_parser.add_argument("directory", type=Path, help="Directory to index")

    # status command
    status_parser = subparsers.add_parser("status", help="Show index status")
    status_parser.add_argument(
        "directory", type=Path, nargs="?", default=Path("."), help="Directory to check"
    )

    # select command (legacy behavior, now explicit)
    select_parser = subparsers.add_parser(
        "select", help="Select newest RAW files and export to text file"
    )
    select_parser.add_argument("input_dir", type=Path, help="Directory to scan")
    select_parser.add_argument(
        "--out", type=Path, required=True, help="Output text file"
    )
    select_parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of files (default: {DEFAULT_N})",
    )
    select_parser.add_argument(
        "--ext",
        default=",".join(RAW_EXTENSIONS),
        help=f"File extensions, comma-separated (default: {','.join(sorted(RAW_EXTENSIONS))})",
    )

    # cull command - detect bursts and mark best-of-burst
    cull_parser = subparsers.add_parser(
        "cull", help="Detect bursts and select best images"
    )
    cull_parser.add_argument("directory", type=Path, help="Indexed directory")
    cull_parser.add_argument(
        "--burst-threshold",
        type=float,
        default=2.0,
        help="Seconds between burst images (default: 2.0)",
    )

    # export command
    export_parser = subparsers.add_parser("export", help="Export selected images")
    export_parser.add_argument("directory", type=Path, help="Indexed directory")
    export_parser.add_argument("--out", type=Path, required=True, help="Output path")
    export_parser.add_argument(
        "--format",
        choices=["list", "copy", "xmp", "gallery"],
        default="list",
        help="Export format: list, copy, xmp, or gallery (default: list)",
    )
    export_parser.add_argument(
        "--top", type=int, default=None, help="Export top N images"
    )
    export_parser.add_argument(
        "--min-sharpness", type=float, default=None, help="Minimum sharpness score"
    )
    export_parser.add_argument(
        "--include-dupes", action="store_true", help="Include duplicate images"
    )
    export_parser.add_argument(
        "--all-burst",
        action="store_true",
        help="Include all burst images, not just best",
    )
    export_parser.add_argument(
        "--rating", type=int, default=5, help="XMP rating (0-5, default: 5)"
    )
    export_parser.add_argument(
        "--label", type=str, default=None, help="XMP color label"
    )

    # score command - compute exhibition scores
    score_parser = subparsers.add_parser(
        "score", help="Compute exhibition scores for images"
    )
    score_parser.add_argument("directory", type=Path, help="Indexed directory")
    score_parser.add_argument(
        "--top", type=int, default=10, help="Show top N images (default: 10)"
    )
    score_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed score breakdown"
    )

    args = parser.parse_args(argv)

    if args.command == "index":
        return cmd_index(args.directory)
    if args.command == "status":
        return cmd_status(args.directory)
    if args.command == "select":
        return cmd_select(args.input_dir, args.out, args.n, args.ext)
    if args.command == "cull":
        return cmd_cull(args.directory, args.burst_threshold)
    if args.command == "export":
        return cmd_export(
            args.directory,
            args.out,
            args.format,
            args.top,
            args.min_sharpness,
            not args.include_dupes,
            not args.all_burst,
            args.rating,
            args.label,
        )
    if args.command == "score":
        return cmd_score(args.directory, args.top, args.verbose)

    parser.print_help()
    return 1


def cmd_select(input_dir: Path, out: Path, n: int, ext: str) -> int:
    """Select newest RAW files and export to text file."""
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        return 1

    extensions = _parse_extensions(ext)
    files = find_raw_files(input_dir, extensions)
    selected = select_newest(files, n)
    write_export(out, selected, input_dir, n)

    print(f"Wrote {len(selected)} paths to {out}")
    return 0


def cmd_index(directory: Path) -> int:
    """Index a directory of RAW files."""
    from qrate.analyze import (
        compute_exposure_score,
        compute_file_hash,
        compute_perceptual_hash,
        compute_sharpness,
        estimate_noise,
    )
    from qrate.db import ImageRecord, QualityScores, get_db
    from qrate.ingest import extract_exif, extract_preview, get_preview_dir

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1

    db = get_db(directory)
    preview_dir = get_preview_dir(directory)
    files = find_raw_files(directory)

    # Check which files need indexing
    paths_with_mtime = [(str(p.resolve()), p.stat().st_mtime) for p in files]
    needs_index = set(db.get_images_needing_index(paths_with_mtime))

    indexed = 0
    skipped = 0
    
    total = len(needs_index)
    if total == 0:
        print(f"All {len(files)} files already indexed")
        print(f"Total: {db.count_images()} files in index")
        return 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=None,
    ) as progress:
        task = progress.add_task("[cyan]Indexing files...", total=total)
        
        for path in files:
            resolved = str(path.resolve())
            if resolved not in needs_index:
                skipped += 1
                continue

            progress.update(task, description=f"[cyan]Processing {path.name}...")
            stat = path.stat()

            # Extract EXIF
            exif = extract_exif(path)

            # Extract preview
            preview_path = extract_preview(path, preview_dir)

            # Compute hashes
            file_hash = compute_file_hash(path)

            # Compute quality scores and perceptual hash from preview (faster)
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
                path=resolved,
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
            db.upsert_image(record)

            # Store quality scores
            if sharpness is not None:
                db.upsert_quality(
                    QualityScores(
                        path=resolved,
                        sharpness=sharpness,
                        exposure_score=exposure,
                        noise_estimate=noise,
                    )
                )

            indexed += 1
            progress.advance(task)

    print(f"\n✓ Indexed {indexed} files, skipped {skipped} unchanged")
    print(f"Total: {db.count_images()} files in index")
    return 0


def cmd_status(directory: Path) -> int:
    """Show index status."""
    from qrate.db import get_db

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1

    db = get_db(directory)
    total = db.count_images()
    dupes = db.count_duplicates()
    groups = db.get_burst_groups()

    print(f"Index: {db.db_path}")
    print(f"Total images: {total}")
    print(f"Exact duplicates: {dupes}")
    print(f"Burst groups: {len(groups)}")
    return 0


def cmd_cull(directory: Path, burst_threshold: float) -> int:
    """Detect bursts and select best images."""
    from qrate.db import get_db
    from qrate.group import process_bursts

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1

    db = get_db(directory)

    if db.count_images() == 0:
        print("No images indexed. Run 'qrate index' first.", file=sys.stderr)
        return 1

    # Detect and process bursts
    num_bursts = process_bursts(db, burst_threshold)
    print(f"Detected {num_bursts} burst groups")

    # Report duplicates
    dupes = db.find_exact_duplicates()
    if dupes:
        print(f"Found {len(dupes)} groups of exact duplicates")

    return 0


def cmd_export(
    directory: Path,
    out: Path,
    fmt: str,
    n: int | None,
    min_sharpness: float | None,
    exclude_duplicates: bool,
    best_of_burst_only: bool,
    rating: int,
    label: str | None,
) -> int:
    """Export selected images."""
    from qrate.db import get_db
    from qrate.export import (
        export_copy,
        export_gallery,
        export_list,
        export_xmp,
        select_for_export,
    )

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1

    db = get_db(directory)

    if db.count_images() == 0:
        print("No images indexed. Run 'qrate index' first.", file=sys.stderr)
        return 1

    # Gallery format has its own selection logic (uses scoring)
    if fmt == "gallery":
        count = export_gallery(db, out, directory, n=n or 20)
        print(f"Exported {count} images to {out}/")
        print(f"See {out}/scores.txt for rankings")
        return 0

    # Select images for other formats
    paths = select_for_export(
        db,
        n=n,
        min_sharpness=min_sharpness,
        exclude_duplicates=exclude_duplicates,
        best_of_burst_only=best_of_burst_only,
    )

    if not paths:
        print("No images match criteria", file=sys.stderr)
        return 1

    print(f"Selected {len(paths)} images")

    if fmt == "list":
        export_list(paths, out, source_dir=directory)
        print(f"Wrote list to {out}")
    elif fmt == "copy":
        count = export_copy(paths, out, source_dir=directory)
        print(f"Copied {count} files to {out}")
    elif fmt == "xmp":
        count = export_xmp(paths, rating=rating, label=label)
        print(f"Created {count} XMP sidecars")

    return 0


def cmd_score(directory: Path, top: int, verbose: bool) -> int:
    """Compute and display exhibition scores."""
    from pathlib import Path as P

    from qrate.db import get_db
    from qrate.score import (
        ExhibitionScore,
        TechnicalScores,
        score_image,
        score_uniqueness,
    )

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1

    db = get_db(directory)

    if db.count_images() == 0:
        print("No images indexed. Run 'qrate index' first.", file=sys.stderr)
        return 1

    images = db.get_all_images()

    # Collect all perceptual hashes for uniqueness scoring
    all_hashes = [img.hash_perceptual for img in images if img.hash_perceptual]

    results: list[tuple[str, ExhibitionScore]] = []
    
    # Filter to images with previews
    images_with_previews = [
        img for img in images
        if img.preview_path and P(img.preview_path).exists()
    ]
    
    if not images_with_previews:
        print("No images with previews found. Run 'qrate index' first.", file=sys.stderr)
        return 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=None,
    ) as progress:
        task = progress.add_task("[cyan]Scoring images...", total=len(images_with_previews))
        
        for img in images_with_previews:
            preview_path = P(img.preview_path)
            progress.update(task, description=f"[cyan]Scoring {Path(img.path).name}...")

            # Reuse existing technical scores from DB
            quality = db.get_quality(img.path)
            existing_tech = None
            if quality:
                existing_tech = TechnicalScores(
                    sharpness=quality.sharpness or 0,
                    exposure=quality.exposure_score or 0,
                    noise=quality.noise_estimate or 0,
                )

            try:
                score = score_image(preview_path, existing_tech)
                # Add uniqueness score
                if img.hash_perceptual:
                    score.uniqueness = score_uniqueness(img.hash_perceptual, all_hashes)
                results.append((img.path, score))
            except Exception as e:
                print(f"  Warning: failed to score {P(img.path).name}: {e}")
            
            progress.advance(task)

    # Sort by final score
    results.sort(key=lambda x: x[1].final_score, reverse=True)

    # Display results
    print(
        f"{'Rank':<5} {'Score':<8} {'Tech':<6} {'Comp':<6} {'Color':<6} {'Uniq':<6} {'File'}"
    )
    print("-" * 80)

    for i, (path, score) in enumerate(results[:top], 1):
        name = P(path).name
        print(
            f"{i:<5} {score.final_score:>6.1f}  "
            f"{score.technical_score:>5.2f}  {score.composition_score:>5.2f}  "
            f"{score.color_score:>5.2f}  {score.uniqueness:>5.2f}  {name}"
        )

        if verbose:
            t = score.technical
            c = score.composition
            col = score.color
            print(
                f"       Technical: sharp={t.sharpness:.0f} subj_sharp={t.subject_sharpness:.2f} exp={t.exposure:.2f} noise={t.noise:.2f}"
            )
            print(
                f"       Composition: thirds={c.thirds_alignment:.2f} balance={c.balance:.2f} simple={c.simplicity:.2f} obstruct={c.obstruction:.2f} clarity={c.subject_clarity:.2f}"
            )
            print(
                f"       Color: harmony={col.harmony:.2f} sat={col.saturation_balance:.2f} contrast={col.color_contrast:.2f}"
            )
            print()

    print()
    print(f"Top {min(top, len(results))} of {len(results)} scored images")

    return 0


if __name__ == "__main__":
    sys.exit(main())
