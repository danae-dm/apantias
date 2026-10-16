# pyright: reportUnusedFunction=false
# Helpers in this module are imported by apantias.io.zarr; the underscore marks
# them as package-internal rather than module-private.

"""HDF5 read helpers."""

from pathlib import Path
from typing import Any, cast

import dask.array as da
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt


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


def print_h5_contents(h5_path: Path | str) -> None:
    """Print all groups and datasets in an HDF5 file with sizes in MB."""
    h5_path = Path(h5_path)
    file_size_mb = h5_path.stat().st_size / 1024**2
    print(f"{h5_path} (file size: {file_size_mb:.2f} MB)")

    with h5py.File(h5_path, "r") as f:

        def walk(group: h5py.Group, depth: int) -> None:
            # Recursive walk instead of visititems: visititems skips empty
            # groups, and the previous root-listing + visititems combination
            # printed root-level entries twice.
            for key in group:  # type: ignore[reportUnknownVariableType]
                obj = group[key]
                indent = "  " * (depth + 1)
                if isinstance(obj, h5py.Dataset):
                    size_mb: float = obj.nbytes / 1024**2  # type: ignore[reportUnknownMemberType]
                    print(
                        f"{indent}{key} [dataset] shape={obj.shape} "  # type: ignore[reportUnknownMemberType]
                        f"dtype={obj.dtype} size={size_mb:.2f} MB"  # type: ignore[reportUnknownMemberType]
                    )
                else:
                    print(f"{indent}{key}/ [group]")
                    walk(obj, depth + 1)  # type: ignore[reportUnknownArgumentType]

        walk(f, 0)


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
