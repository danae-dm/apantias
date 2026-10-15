"""Generic Dask array reductions."""

# dask.array.map_blocks is only partially typed by its stubs; the runtime contract
# is a dask Array, so strict unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt


def apply_pixelwise(data: da.Array, path: str | Path, func: Callable[[npt.NDArray[Any]], Any]) -> None:
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

    def _apply_and_squeeze(chunk: npt.NDArray[Any]) -> Any:
        return np.asarray(func(chunk.squeeze()))

    result = da.map_blocks(_apply_and_squeeze, data, dtype=float, drop_axis=[1, 2])
    result.to_zarr(path)
