"""Statistical and generic reductions on Dask arrays."""

from .materialize import Materializer
from .rechunk import rechunk_to_pixels
from .reductions import apply_pixelwise
from .stats import (
    compute_common_modes,
    compute_median,
    compute_msd,
    compute_offset_corr,
    compute_signals_mean,
    compute_slopes,
    subtract,
)

__all__ = [
    "Materializer",
    "apply_pixelwise",
    "compute_common_modes",
    "compute_median",
    "compute_msd",
    "compute_offset_corr",
    "compute_signals_mean",
    "compute_slopes",
    "rechunk_to_pixels",
    "subtract",
]
