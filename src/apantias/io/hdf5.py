# Helpers in this module are imported by apantias.io.zarr; the underscore marks
# them as package-internal rather than module-private.
# pyright: reportUnusedFunction=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""HDF5 read helpers and the final analysis-output archive writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import dask.array as da
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import zarr

from apantias.inspect.h5 import print_h5_contents
from apantias.io.outputs import Pipeline

__all__ = [
    "load_h5_dataset",
    "print_h5_contents",
    "split_h5_dataset_path",
    "write_analysis_outputs_to_h5",
]


def _read_h5_batch(
    h5_path: Path,
    dataset_name: str,
    batch_start: int,
    batch_end: int,
    frame_indices: list[int] | None = None,
    nreps_indices: list[int] | None = None,
) -> npt.NDArray[Any]:
    """Read one batch of frames from an HDF5 dataset as a NumPy array."""
    with h5py.File(h5_path, "r") as f:
        ds = cast(h5py.Dataset, f[dataset_name])
        # h5py has no type stubs; cast the slice result to the expected ndarray.
        batch: npt.NDArray[Any] = np.asarray(ds[batch_start:batch_end])  # type: ignore[index]

    # Translate frame indices from original-axis → batch-relative.
    if frame_indices is not None:
        batch_frame_indices = [i - batch_start for i in frame_indices if 0 <= i - batch_start < batch.shape[0]]
        if batch_frame_indices:
            batch = batch[batch_frame_indices]

    # Nreps indices are applied directly (no translation needed).
    if nreps_indices is not None:
        batch = batch[:, :, nreps_indices, :]

    return batch


def split_h5_dataset_path(ext_offset: Path | str) -> tuple[Path, str] | None:
    """Split ``<file.h5>/<internal dataset path>`` into (file, dataset path).

    Returns None when *ext_offset* does not reference an HDF5 file (e.g. a
    zarr store path). The dataset part may be empty, in which case the file
    is expected to contain exactly one dataset (auto-detected on load).
    """
    s = str(ext_offset)
    lower = s.lower()
    for ext in (".hdf5", ".h5"):
        idx = lower.find(ext)
        if idx != -1:
            file_path = Path(s[: idx + len(ext)])
            dataset_name = s[idx + len(ext) :].strip("/")
            return file_path, dataset_name
    return None


def load_h5_dataset(file_path: Path, dataset_name: str | None = None) -> da.Array:
    """Load an HDF5 dataset fully into memory as a dask array.

    Offset maps are tiny (n_rows, n_cols), so eager reading is cheap and keeps
    h5py objects out of the Dask graph: distributed workers receive the plain
    ndarray instead of an unpicklable h5py dataset handle. Pass *dataset_name*
    to select a specific dataset; with None, the file must contain exactly one
    dataset (auto-detected via _find_h5_dataset).
    """
    with h5py.File(file_path, "r") as f:
        try:
            ds = cast(h5py.Dataset, f[dataset_name] if dataset_name else f[_find_h5_dataset(f)])
        except KeyError as exc:
            raise ValueError(f"Dataset '{dataset_name}' not found in {file_path}") from exc
        data: npt.NDArray[Any] = np.asarray(ds[...])  # type: ignore[index]
    return cast(da.Array, da.from_array(data, chunks=-1))  # type: ignore[reportArgumentType]


def _find_h5_dataset(f: h5py.File) -> str:
    """Return the single dataset path in an HDF5 file."""
    datasets: list[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    # h5py has no type stubs for visititems; the callback signature is correct at runtime.
    f.visititems(visit)  # type: ignore[reportUnknownMemberType]
    if len(datasets) != 1:
        raise ValueError(f"Expected exactly one dataset in {f.filename}, found {len(datasets)}: {datasets}")
    return datasets[0]


def write_analysis_outputs_to_h5(pipeline: Pipeline, h5_path: Path) -> None:
    """Write the archived Zarr outputs into a single HDF5 archive file.

    The archive step is driven entirely by the ``Pipeline`` registry.  Each
    stage knows whether it should be archived, which chunk/compression strategy
    to use, and what its description is.  The description is stored as an HDF5
    attribute so the archive documents itself.
    """
    # Split archived stages by their archive strategy.
    small_stages = [s for s in pipeline.archived() if s.archive_strategy == "small"]
    large_stages = [s for s in pipeline.archived() if s.archive_strategy == "large"]

    with h5py.File(h5_path, "w") as f:
        # Small outputs: read fully into memory and write contiguously with gzip.
        for stage in small_stages:
            data = np.asarray(zarr.open_array(str(stage.path), mode="r"))
            ds = f.create_dataset(stage.name, data=data, compression="gzip")
            ds.attrs["description"] = stage.description

        # Large outputs: stream by source Zarr chunks and write with lzf.
        for stage in large_stages:
            arr = zarr.open_array(str(stage.path), mode="r")

            # Align HDF5 chunks with the source Zarr chunks along the frame
            # axis so each Zarr chunk maps to one HDF5 write operation.  A
            # fixed ``archive_chunk_frames`` on the stage overrides the source
            # chunk size; otherwise inherit it from Zarr.  Cap by n_frames so
            # small arrays do not get oversized chunks.
            frame_chunk_size = stage.archive_chunk_frames or arr.chunks[0]
            frame_chunk_size = min(frame_chunk_size, arr.shape[0])

            ds = f.create_dataset(
                stage.name,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=(frame_chunk_size, *arr.shape[1:]),
                compression="lzf",
            )
            ds.attrs["description"] = stage.description

            for batch_start in range(0, arr.shape[0], frame_chunk_size):
                batch_end = min(batch_start + frame_chunk_size, arr.shape[0])
                ds[batch_start:batch_end] = np.asarray(arr[batch_start:batch_end])
