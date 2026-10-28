"""Read-only inspection helpers for generated analysis outputs."""

from .h5 import (
    print_archive_summary,
    print_dataset_info,
    print_dataset_stats,
    print_h5_contents,
)

__all__ = [
    "print_archive_summary",
    "print_dataset_info",
    "print_dataset_stats",
    "print_h5_contents",
]
