"""Tests for archiving analysis outputs to HDF5."""

# h5py has no type stubs; File.__getitem__ returns a Group|Dataset|Datatype
# union, so dataset access is cast to h5py.Dataset like in the src code.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from pathlib import Path
from typing import cast

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import zarr

from apantias.archive import archive
from apantias.settings import AnalysisSettings, AppSettings


def test_archive_reads_stored_zarr(tmp_path: Path) -> None:
    """Archive must read outputs from their stored zarr locations, not recompute."""
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    out_path = tmp_path / "median"
    arr = zarr.open_array(str(out_path), mode="w", shape=data.shape, chunks=(2, 2), dtype=data.dtype)
    arr[:] = data

    config = AppSettings(analysis=AnalysisSettings(h5_archive=tmp_path / "archive.h5"))
    archive(config, [("median", out_path)])

    with h5py.File(tmp_path / "archive.h5", "r") as f:
        ds = cast(h5py.Dataset, f["median"])
        np.testing.assert_allclose(cast(npt.NDArray[np.float64], ds[:]), data)
        assert "settings" in f


def test_archive_overwrites_existing_dataset(tmp_path: Path) -> None:
    """Re-running with an existing archive replaces the datasets instead of failing."""
    first = np.zeros((2, 2))
    second = np.ones((2, 2))
    out_path = tmp_path / "msd"
    arr = zarr.open_array(str(out_path), mode="w", shape=second.shape, chunks=(2, 2), dtype="f8")

    config = AppSettings(analysis=AnalysisSettings(h5_archive=tmp_path / "archive.h5"))

    arr[:] = first
    archive(config, [("msd", out_path)])
    arr[:] = second
    archive(config, [("msd", out_path)])

    with h5py.File(tmp_path / "archive.h5", "r") as f:
        ds = cast(h5py.Dataset, f["msd"])
        np.testing.assert_allclose(cast(npt.NDArray[np.float64], ds[:]), second)
