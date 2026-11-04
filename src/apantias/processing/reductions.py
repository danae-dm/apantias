"""Generic Dask array reductions."""

# dask.array.map_blocks is only partially typed by its stubs; the runtime contract
# is a dask Array, so strict unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt


def apply_pixelwise(
    data: da.Array,
    func: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
) -> da.Array:
    """Apply a custom function to each pixel's time series.

    Parameters
    ----------
    data : da.Array
        Pixelwise data of shape (n_frames, n_pixels), chunked as (n_frames, 1)
        - one chunk per pixel's time series.
    func : callable
        User function that takes a pixel's (n_frames,) time series and
        returns an array of values (e.g., fitted parameters). Applied independently
        to each pixel without loading the full array into memory.

    Returns
    -------
    da.Array
        Output array of shape (n_pixels * k,) where k is the function's output dimension.
        Reshape to (n_pixels, k) then to (n_rows, n_cols, k) as needed.

    Notes
    -----
    This function processes each pixel's time series independently.
    For 3D input (n_frames, n_rows, n_cols), reshape to (n_frames, n_rows * n_cols)
    and chunk as (n_frames, 1) before calling this function. After processing,
    reshape the output:

        result = apply_pixelwise(d, func)
        result = result.reshape(n_rows * n_cols, k)
        result = result.reshape(n_rows, n_cols, k)
    """

    def _apply_to_pixel(chunk: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if chunk.size == 0:
            return np.empty((1,), dtype=np.float64)
        series = chunk.squeeze()  # (n_frames,) after squeezing singleton dim
        result = np.asarray(func(series))
        return result

    # 2D input: (n_frames, n_pixels) with chunks (n_frames, 1)
    if len(data.shape) != 2:
        raise ValueError(f"Input must be 2D, got {len(data.shape)}D")

    # Drop axis 0 (frames), output is flattened (n_pixels * k,)
    result = da.map_blocks(  # type: ignore[reportUnknownMemberType]
        _apply_to_pixel,
        data,
        dtype=float,
        drop_axis=0,
    )
    return result


__all__: list[str] = ["apply_pixelwise"]
