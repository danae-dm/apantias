"""Array rechunking transformations."""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import zarr
from dask import delayed  # pyright: ignore
from dask.base import compute

from apantias.io.zarr import _blosc_codecs

_logger = logging.getLogger(__name__)


def _rechunk_col_batch(
    source_store: str,
    source_array_path: str,
    target_store: str,
    target_array_path: str,
    col: int,
    col_batch: int,
    n_col: int,
    n_row: int,
) -> None:
    """Worker: read a ``col_batch``-wide column band, write single-pixel chunks."""
    source = zarr.open_array(source_store, path=source_array_path, mode="r")
    target = zarr.open_array(target_store, path=target_array_path, mode="r+")

    col_end = min(col + col_batch, n_col)
    band = np.asarray(source[:, col:col_end, :, :])  # (n_frames, col_batch, n_reps, n_row)

    for c in range(col_end - col):
        for j in range(n_row):
            # Each write targets exactly one pixel chunk -> conflict-free.
            target[:, col + c, :, j] = band[:, c, :, j]


def rechunk_to_pixels(
    source_path: str | Path,
    target_path: str | Path,
    col_batch: int = 1,
) -> None:
    """
    Rechunks a zarr store so that each chunk holds all frames and all readouts
    for a single spatial pixel. The target chunk shape is
    ``(n_frames, 1, n_reps, 1)`` — i.e. one chunk per (column, row) pixel.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    # Parse source path to extract store and array path
    source_str = str(source_path)
    if ".zarr" in source_str:
        store_end = source_str.find(".zarr") + len(".zarr")
        source_store = source_str[:store_end]
        source_array_path = source_str[store_end:].lstrip("/")
        source = cast(zarr.Array, zarr.open_array(source_store, path=source_array_path, mode="r"))  # type: ignore
    else:
        source_store = source_str
        source_array_path = ""
        source = cast(zarr.Array, zarr.open(source_str, mode="r"))

    n_frames, n_col, n_reps, n_row = source.shape

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
        )  # type: ignore
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

    tasks = []
    for i in range(0, n_col, col_batch):
        task = delayed(_rechunk_col_batch)(
            source_store, source_array_path, target_store, target_array_path, i, col_batch, n_col, n_row
        )
        tasks.append(task)

    _logger.info("Executing %d Dask tasks to rechunk %s...", len(tasks), source_path)
    compute(*tasks)

    _logger.info("Rechunked %s -> %s", source_path, target_path)
