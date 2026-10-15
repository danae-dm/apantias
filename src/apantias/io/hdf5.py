# pyright: reportUnusedFunction=false
# Helpers in this module are imported by apantias.io.zarr; the underscore marks
# them as package-internal rather than module-private.

"""HDF5 read helpers."""

from pathlib import Path
from typing import Any, cast

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
