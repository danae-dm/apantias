"""Generic Dask array reductions."""

# dask.array.map_blocks is only partially typed by its stubs; the runtime contract
# is a dask Array, so strict unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt


def apply_pixelwise(data: da.Array, func: Callable[[npt.NDArray[Any]], Any]) -> da.Array:
    """Apply a custom function to each pixel's time series.

    Parameters
    ----------
    data : da.Array
        Pixelwise data of shape (n_frames, n_cols, n_reps, n_rows), chunked
        as (n_frames, 1, n_reps, 1) — one chunk per spatial pixel, so each
        chunk holds one pixel's complete (n_frames, n_reps) time series.
    func : callable
        User function that takes a pixel's (n_frames, n_reps) time series and
        returns a single scalar value. Applied independently to each pixel
        without loading the full array into memory.

    Returns
    -------
    da.Array
        Output of shape (n_cols, n_rows) — one value per spatial pixel.

    Notes
    -----
    Dask's ``map_blocks`` with ``drop_axis`` rechunks the dropped axes to a
    single chunk, so the frame and rep axes must each have one chunk (as the
    pixel-chunked store does). The frame and rep axes are dropped, keeping
    one output chunk of shape (1, 1) per (col, row) pixel. Dask also probes
    the block function with an empty array to infer metadata; the wrapper
    short-circuits that probe so user functions like ``max``/``min`` do not
    raise on empty input.
    """

    def _apply_to_pixel(chunk: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if chunk.size == 0:
            # Metadata probe with an empty array: return an empty (1, 1) block.
            return np.empty((1, 1), dtype=np.float64)
        series = chunk.squeeze()
        return np.asarray(func(series)).reshape(1, 1)

    result = da.map_blocks(_apply_to_pixel, data, dtype=float, drop_axis=[0, 2])  # type: ignore[reportUnknownMemberType]
    return result
