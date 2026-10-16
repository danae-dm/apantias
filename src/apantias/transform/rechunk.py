# This module is a thin orchestration layer over Zarr and Dask, both of which ship
# incomplete type stubs. We relax strict "unknown" checks here rather than
# cluttering every library call with an ignore comment.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""Array rechunking transformations."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from dask import delayed  # pyright: ignore
from dask.base import compute

from apantias.io.zarr import _blosc_codecs  # pyright: ignore[reportPrivateUsage]

_logger = logging.getLogger(__name__)


def _rechunk_row_batch(
    source_store: str,
    source_array_path: str,
    target_store: str,
    target_array_path: str,
    row: int,
    row_batch: int,
    n_rows: int,
    n_cols: int,
) -> None:
    """Worker: read a ``row_batch``-wide row band, write single-pixel chunks."""
    source = zarr.open_array(source_store, path=source_array_path, mode="r")
    target = zarr.open_array(target_store, path=target_array_path, mode="r+")

    row_end = min(row + row_batch, n_rows)
    band = np.asarray(source[:, row:row_end, :, :])  # (n_frames, row_batch, n_reps, n_cols)

    for r in range(row_end - row):
        for j in range(n_cols):
            # Each write targets exactly one pixel chunk -> conflict-free.
            target[:, row + r, :, j] = band[:, r, :, j]


def rechunk_to_pixels(
    source_path: str | Path,
    target_path: str | Path,
    row_batch: int = 1,
) -> None:
    """
    Rechunks a zarr store so that each chunk holds all frames and all readouts
    for a single spatial pixel. The target chunk shape is
    ``(n_frames, 1, n_reps, 1)`` — i.e. one chunk per (row, column) pixel.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    # Parse source path to extract store and array path
    source_str = str(source_path)
    if ".zarr" in source_str:
        store_end = source_str.find(".zarr") + len(".zarr")
        source_store = source_str[:store_end]
        source_array_path = source_str[store_end:].lstrip("/")
        source = zarr.open_array(source_store, path=source_array_path, mode="r")
    else:
        source_store = source_str
        source_array_path = ""
        source = zarr.open_array(source_str, mode="r")

    n_frames, n_rows, n_reps, n_cols = source.shape

    # Parse target path to extract store and array path
    target_str = str(target_path)
    if ".zarr" in target_str:
        store_end = target_str.find(".zarr") + len(".zarr")
        target_store = target_str[:store_end]
        target_array_path = target_str[store_end:].lstrip("/")
        # create empty array with single-pixel chunks
        zarr.open_array(
            target_store,
            path=target_array_path,
            mode="w",
            shape=source.shape,
            chunks=(n_frames, 1, n_reps, 1),
            dtype=source.dtype,
            zarr_format=3,
            codecs=_blosc_codecs(),
        )
    else:
        target_store = target_str
        target_array_path = ""
        # create empty array with single-pixel chunks
        zarr.open_array(
            target_store,
            mode="w",
            shape=source.shape,
            chunks=(n_frames, 1, n_reps, 1),
            dtype=source.dtype,
            zarr_format=3,
            codecs=_blosc_codecs(),
        )

    tasks: list[Any] = []
    for i in range(0, n_rows, row_batch):
        task = delayed(_rechunk_row_batch)(
            source_store, source_array_path, target_store, target_array_path, i, row_batch, n_rows, n_cols
        )
        tasks.append(task)

    _logger.info("Executing %d Dask tasks to rechunk %s...", len(tasks), source_path)
    compute(*tasks)

    _logger.info("Rechunked %s -> %s", source_path, target_path)
