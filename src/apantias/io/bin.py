"""Binary file parsing and conversion to HDF5."""

# h5py ships without type stubs and dask.base.compute is only partially typed,
# so strict-mode "unknown type" checks are relaxed in this binary-conversion module.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import logging
from pathlib import Path
from typing import Any

import h5py  # pyright: ignore[reportMissingTypeStubs]
import hdf5plugin
import numpy as np
import numpy.typing as npt
from dask import delayed  # pyright: ignore
from dask.base import compute  # type: ignore[reportUnknownVariableType]

from apantias.settings import FrameSettings

# These helpers live in a dedicated helper module and are imported here; the
# underscore indicates "package-internal" rather than "module-private".
from ._helpers import _frame_chunk_size, _parse_bin_frames, _read_frame_batch  # pyright: ignore[reportPrivateUsage]

_logger = logging.getLogger(__name__)


def bin_to_h5(
    bin_path: str | Path,
    nreps: int,
    h5_path: str | Path,
    dataset_name: str,
    frame: FrameSettings,
    offset: int = 8,
    key_ints: int = 1,
) -> Path:
    """Write the frames of a binary file to a compressed HDF5 dataset."""
    bin_path = Path(bin_path)
    h5_path = Path(h5_path)

    frame_start_indices, frame_end_indices, n_frames = _parse_bin_frames(
        bin_path, offset, nreps, frame.cols, frame.rows, key_ints
    )
    _logger.info("Found %d valid frames in %s", n_frames, bin_path)

    chunk_size = _frame_chunk_size(nreps, frame.cols, frame.rows, frame.target_chunk_bytes)
    filters = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=1)
    shape = (n_frames, frame.cols, nreps, frame.rows)
    chunk_shape = (chunk_size, frame.cols, nreps, frame.rows)

    # Keep all reps (no slice)
    nreps_slice = slice(None)

    with h5py.File(h5_path, "w") as f:
        # h5py has no type stubs; the dataset handle is treated as Any.
        ds: Any = f.create_dataset(  # type: ignore[reportUnknownMemberType]
            dataset_name, shape=shape, dtype="uint16", chunks=chunk_shape, **filters
        )
        ds.attrs["nreps"] = nreps
        ds.attrs["offset"] = offset
        ds.attrs["n_frames"] = n_frames
        ds.attrs["column_size"] = frame.cols
        ds.attrs["row_size"] = frame.rows
        ds.attrs["raw_row_size"] = frame.rows + key_ints

        # Build delayed tasks — one per batch
        delayed_tasks: list[Any] = []
        for batch_start in range(0, n_frames, chunk_size):
            batch_end = min(batch_start + chunk_size, n_frames)
            delayed_tasks.append(
                delayed(_write_h5_batch)(
                    ds,
                    batch_start,
                    batch_end,
                    bin_path,
                    offset,
                    frame_start_indices,
                    frame_end_indices,
                    nreps,
                    nreps_slice,
                    frame.cols,
                    frame.rows,
                    frame.rows + key_ints,
                )
            )

        # Execute in parallel (threads are fine here — GIL released by
        # numpy/hdf5plugin under the hood)
        compute(*delayed_tasks, scheduler="threads")

    return h5_path


def _write_h5_batch(
    ds: Any,
    start: int,
    end: int,
    bin_path: Path,
    offset: int,
    frame_start_indices: npt.NDArray[np.intp],
    frame_end_indices: npt.NDArray[np.intp],
    nreps: int,
    nreps_slice: slice,
    cols: int,
    rows: int,
    raw_row_size: int,
) -> None:
    """Read one batch from binary and write it to a pre-opened h5py dataset."""
    batch = _read_frame_batch(
        bin_path,
        offset,
        frame_start_indices,
        frame_end_indices,
        start,
        end,
        nreps,
        nreps_slice,
        cols,
        rows,
        raw_row_size,
    )
    ds[start:end] = batch
