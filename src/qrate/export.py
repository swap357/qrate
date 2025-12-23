"""Export module: output selected images in various formats."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

if TYPE_CHECKING:
    from qrate.db import Database

ExportFormat = Literal["list", "copy", "xmp", "gallery"]


def export_list(
    paths: list[str],
    out_path: Path,
    source_dir: Path | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    """Export paths to a text file.

    Args:
        paths: List of absolute paths to export.
        out_path: Output file path.
        source_dir: Original source directory (for header).
        metadata: Optional metadata to include in header.
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# qrate export",
        f"# generated: {timestamp}",
        f"# count: {len(paths)}",
    ]

    if source_dir:
        lines.append(f"# source: {source_dir}")

    if metadata:
        for key, value in metadata.items():
            lines.append(f"# {key}: {value}")

    lines.append("")
    lines.extend(paths)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_copy(
    paths: list[str],
    out_dir: Path,
    preserve_structure: bool = False,
    source_dir: Path | None = None,
) -> int:
    """Copy files to output directory.

    Args:
        paths: List of absolute paths to copy.
        out_dir: Destination directory.
        preserve_structure: If True, preserve subdirectory structure.
        source_dir: Base directory for structure preservation.

    Returns:
        Number of files copied.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for path_str in paths:
        src = Path(path_str)
        if not src.exists():
            continue

        if preserve_structure and source_dir:
            try:
                rel = src.relative_to(source_dir)
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
            except ValueError:
                dest = out_dir / src.name
        else:
            dest = out_dir / src.name

        # Handle name collisions
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = dest.with_name(f"{stem}_{counter}{suffix}")
                counter += 1

        shutil.copy2(src, dest)
        copied += 1

    return copied


def export_xmp(
    paths: list[str],
    rating: int = 5,
    label: str | None = None,
    keywords: list[str] | None = None,
) -> int:
    """Create XMP sidecar files for Lightroom/Bridge compatibility.

    Args:
        paths: List of absolute paths to create XMP sidecars for.
        rating: Star rating (0-5).
        label: Color label (Red, Yellow, Green, Blue, Purple).
        keywords: Keywords to add.

    Returns:
        Number of XMP files created.
    """
    created = 0

    for path_str in paths:
        src = Path(path_str)
        if not src.exists():
            continue

        xmp_path = src.with_suffix(src.suffix + ".xmp")

        # Don't overwrite existing XMP files
        if xmp_path.exists():
            continue

        xmp_content = _generate_xmp(rating, label, keywords)
        xmp_path.write_text(xmp_content, encoding="utf-8")
        created += 1

    return created


def _generate_xmp(
    rating: int = 5,
    label: str | None = None,
    keywords: list[str] | None = None,
) -> str:
    """Generate XMP sidecar content.

    Args:
        rating: Star rating (0-5).
        label: Color label.
        keywords: Keywords list.

    Returns:
        XMP file content as string.
    """
    # Minimal XMP that Lightroom/Bridge can read
    keyword_xml = ""
    if keywords:
        kw_items = "\n".join(f"        <rdf:li>{kw}</rdf:li>" for kw in keywords)
        keyword_xml = f"""
      <dc:subject>
       <rdf:Bag>
{kw_items}
       </rdf:Bag>
      </dc:subject>"""

    label_xml = f'\n      xmp:Label="{label}"' if label else ""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="qrate">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmp:Rating="{rating}"{label_xml}>{keyword_xml}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def select_for_export(
    db: "Database",
    n: int | None = None,
    min_sharpness: float | None = None,
    exclude_duplicates: bool = True,
    best_of_burst_only: bool = True,
) -> list[str]:
    """Select images for export based on criteria.

    Args:
        db: Database instance.
        n: Maximum number of images to export.
        min_sharpness: Minimum sharpness score.
        exclude_duplicates: Exclude exact duplicates (keep first).
        best_of_burst_only: Only include best image from each burst.

    Returns:
        List of paths to export.
    """
    from qrate.db import Database

    assert isinstance(db, Database)

    # Start with all images
    all_images = db.get_all_images()
    paths = [img.path for img in all_images]

    # Exclude duplicates
    if exclude_duplicates:
        dupe_groups = db.find_exact_duplicates()
        # Keep first of each dupe group, exclude rest
        to_exclude = set()
        for group in dupe_groups:
            to_exclude.update(group[1:])
        paths = [p for p in paths if p not in to_exclude]

    # Apply burst filtering
    if best_of_burst_only:
        burst_groups = db.get_burst_groups()
        in_burst = set()
        best_of = set()

        for members in burst_groups.values():
            for m in members:
                in_burst.add(m.path)
                if m.is_best:
                    best_of.add(m.path)

        # Keep standalone images + best of bursts
        paths = [p for p in paths if p not in in_burst or p in best_of]

    # Filter by sharpness
    if min_sharpness is not None:
        filtered = []
        for path in paths:
            quality = db.get_quality(path)
            if (
                quality
                and quality.sharpness is not None
                and quality.sharpness >= min_sharpness
            ):
                filtered.append(path)
            elif quality is None:
                # No quality data, include by default
                filtered.append(path)
        paths = filtered

    # Sort by quality score (best first)
    def score(path: str) -> float:
        quality = db.get_quality(path)
        if quality:
            return (quality.sharpness or 0) + (quality.exposure_score or 0) * 100
        return 0

    paths.sort(key=score, reverse=True)

    # Limit count
    if n is not None:
        paths = paths[:n]

    return paths


def export_gallery(
    db: "Database",
    out_dir: Path,
    source_dir: Path,
    n: int = 20,
) -> int:
    """Export best images as a gallery with RAW files, JPG previews, and scores.

    Creates:
        out_dir/
            raw/
                001_DSC_0001.NEF
            jpg/
                001_DSC_0001.jpg
            scores.txt

    Args:
        db: Database instance.
        out_dir: Output directory (created if needed).
        source_dir: Source directory (for finding previews).
        n: Number of images to export.

    Returns:
        Number of images exported.
    """
    from qrate.score import score_image

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    jpg_dir = out_dir / "jpg"
    raw_dir.mkdir(exist_ok=True)
    jpg_dir.mkdir(exist_ok=True)

    preview_dir = source_dir / ".qrate_previews"

    # Get images and score them
    all_images = db.get_all_images()
    images_with_previews = [
        img for img in all_images
        if (preview_dir / f"{Path(img.path).stem}_preview.jpg").exists()
    ]
    
    scored = []
    
    if images_with_previews:
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
                path = Path(img.path)
                preview_path = preview_dir / f"{path.stem}_preview.jpg"
                progress.update(task, description=f"[cyan]Scoring {path.name}...")
                
                try:
                    s = score_image(preview_path)
                    scored.append((img.path, path.name, preview_path, s))
                except Exception:
                    pass
                
                progress.advance(task)

    # Sort by score
    scored.sort(key=lambda x: x[3].final_score, reverse=True)
    scored = scored[:n]

    # Export
    exported = 0
    
    if scored:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=None,
        ) as progress:
            task = progress.add_task("[cyan]Exporting gallery...", total=len(scored))
            
            score_lines = [
                "# qrate gallery export",
                f"# source: {source_dir}",
                f"# count: {len(scored)}",
                f"# generated: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
                "",
                "Rank  Score  File",
                "-" * 60,
            ]

            for rank, (raw_path, name, preview_path, s) in enumerate(scored, 1):
                progress.update(task, description=f"[cyan]Exporting {name}...")
        raw_src = Path(raw_path)
        stem = raw_src.stem
        suffix = raw_src.suffix

                # Copy RAW file
                raw_out = raw_dir / f"{rank:03d}_{stem}{suffix}"
                if raw_src.exists():
                    shutil.copy2(raw_src, raw_out)

                # Copy JPG preview
                jpg_out = jpg_dir / f"{rank:03d}_{stem}.jpg"
                shutil.copy2(preview_path, jpg_out)

                exported += 1

                # Add to scores with aggregated breakdown
                tech_score = s.technical_score
                comp_score = s.composition_score
                color_score = s.color_score
                uniq_score = s.uniqueness

                # Calculate weighted contributions
                w_tech = s.WEIGHT_TECHNICAL
                w_comp = s.WEIGHT_COMPOSITION
                w_color = s.WEIGHT_COLOR
                w_uniq = s.WEIGHT_UNIQUENESS

                # Check if obstruction penalty applies
                obstruction_penalty = 1.0 - s.composition.obstruction
                if obstruction_penalty > 0.15:
                    shift = obstruction_penalty * 0.15
                    w_tech = max(0.15, w_tech - shift)
                    w_comp = min(0.45, w_comp + shift)

                tech_contrib = tech_score * w_tech * 100
                comp_contrib = comp_score * w_comp * 100
                color_contrib = color_score * w_color * 100
                uniq_contrib = uniq_score * w_uniq * 100

                score_lines.append(f"{rank:3d}   {s.final_score:5.1f}  {name}")
                score_lines.append(
                    f"      Technical:   {tech_score:.2f} (weight {w_tech:.0%}) → {tech_contrib:.1f} pts"
                )
                score_lines.append(
                    f"      Composition: {comp_score:.2f} (weight {w_comp:.0%}) → {comp_contrib:.1f} pts"
                )
                score_lines.append(
                    f"      Color:       {color_score:.2f} (weight {w_color:.0%}) → {color_contrib:.1f} pts"
                )
                score_lines.append(
                    f"      Uniqueness:  {uniq_score:.2f} (weight {w_uniq:.0%}) → {uniq_contrib:.1f} pts"
                )
                score_lines.append("")
                
                progress.advance(task)

    # Write scores
    scores_path = out_dir / "scores.txt"
    scores_path.write_text("\n".join(score_lines), encoding="utf-8")

    return exported
