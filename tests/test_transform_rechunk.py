"""Tests for apantias.transform.rechunk."""

# Tests exercise Zarr's runtime API; its stubs are incomplete in strict mode.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from pathlib import Path

import numpy as np
import zarr

from apantias.transform.rechunk import rechunk_to_pixels


def test_rechunk_to_pixels(tmp_path: Path) -> None:
    source = tmp_path / "src.zarr"
    target = tmp_path / "dst.zarr"
    data = np.arange(4 * 4 * 3 * 4).reshape(4, 4, 3, 4).astype("uint16")
    zarr.open_array(
        str(source),
        mode="w",
        shape=data.shape,
        chunks=(2, 4, 3, 4),
        dtype="uint16",
        zarr_format=3,
    )
    arr = zarr.open_array(str(source), mode="r+")
    arr[:] = data

    rechunk_to_pixels(source, target, row_batch=2)

    result = zarr.open_array(str(target), mode="r")[:]
    np.testing.assert_array_equal(result, data)
    assert zarr.open_array(str(target), mode="r").chunks == (4, 1, 3, 1)
