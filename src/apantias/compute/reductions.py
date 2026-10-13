"""Generic Dask array reductions."""

from pathlib import Path

import dask.array as da
import numpy as np


def apply_pixelwise(data: da.Array, path: str | Path, func) -> None:
    """Apply a custom function to each pixel's time series along axis 0.

    Parameters
    ----------
    data : da.Array
        Pixelwise data of shape (n_frames, n_rows, n_cols), chunked
        as (n_frames, 1, 1) — one chunk per pixel.
    path : str or Path
        Output zarr path.
    func : callable
        User function that takes a 1-D array of shape (n_frames,) and
        returns a single scalar value. Applied independently to each
        pixel without loading the full array into memory.
    """

    def _apply_and_squeeze(chunk):
        return np.asarray(func(chunk.squeeze()))

    result = da.map_blocks(_apply_and_squeeze, data, dtype=float, drop_axis=[1, 2])
    result.to_zarr(path)
