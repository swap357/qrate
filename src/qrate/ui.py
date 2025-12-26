"""UI utilities for qrate CLI."""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def create_progress() -> Progress:
    """Create a standard progress bar for qrate operations.

    Returns:
        Configured Progress instance with spinner, description,
        bar, percentage, count, and elapsed time columns.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=None,
    )
