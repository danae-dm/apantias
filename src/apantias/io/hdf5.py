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


def _archive_chunk_shape(shape: tuple[int, ...], dtype: np.dtype[Any], threshold_bytes: int) -> tuple[int, ...]:
    """Return an HDF5 chunk shape for *shape*/*dtype* that respects *threshold_bytes*.

    If the total array fits below the threshold, the full array shape is used
    (single chunk).  Otherwise the array is chunked along the first (frame) axis
    so that each chunk is at most *threshold_bytes*.
    """
    if not shape:
        return shape

    bytes_per_frame = int(dtype.itemsize) * int(np.prod(shape[1:]))
    total_bytes = bytes_per_frame * shape[0]
    if total_bytes <= threshold_bytes:
        return shape

    frame_chunk = max(1, threshold_bytes // bytes_per_frame)
    frame_chunk = min(frame_chunk, shape[0])
    return (frame_chunk, *shape[1:])


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


def write_analysis_outputs_to_h5(
    pipeline: Pipeline,
    h5_path: Path,
    *,
    chunk_threshold_bytes: int,
) -> None:
    """Write the archived Zarr outputs into a single HDF5 archive file.

    The archive step is driven by the ``Pipeline`` registry for *which* arrays
    are written, but the chunk/compression layout is decided here from the
    configured ``chunk_threshold_bytes``:

    * Arrays that fit in one chunk below the threshold are written as a single
      chunk with gzip.
    * Larger arrays are chunked along the frame axis (at most
      ``chunk_threshold_bytes`` per chunk) and written with lzf.

    The stage description is stored as an HDF5 attribute so the archive
    documents itself.
    """
    with h5py.File(h5_path, "w") as f:
        for stage in pipeline.archived():
            arr = zarr.open_array(str(stage.path), mode="r")
            chunks = _archive_chunk_shape(arr.shape, arr.dtype, chunk_threshold_bytes)
            compression = "gzip" if chunks == arr.shape else "lzf"

            ds = f.create_dataset(
                stage.name,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=chunks,
                compression=compression,
            )
            ds.attrs["description"] = stage.description

            for batch_start in range(0, arr.shape[0], chunks[0]):
                batch_end = min(batch_start + chunks[0], arr.shape[0])
                ds[batch_start:batch_end] = np.asarray(arr[batch_start:batch_end])
