# Helpers in this module are imported by apantias.io.zarr; the underscore marks
# them as package-internal rather than module-private.
# pyright: reportUnusedFunction=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""HDF5 read helpers and the final analysis-output archive writer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import zarr

if TYPE_CHECKING:
    # Imported lazily to avoid a circular import: apantias.standard imports
    # this module, and AnalysisPaths lives there.
    from apantias.standard import AnalysisPaths


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


def write_analysis_outputs_to_h5(paths: AnalysisPaths, h5_path: Path) -> None:
    """Write the final Zarr outputs into a single HDF5 archive file.

    This is a single final pass over the materialized Zarr stores (called
    after all pipeline stages have completed), not incremental writing: a
    one-shot archive keeps the Zarr working stores as the single source of
    truth during the run and avoids partial archives if a stage fails.

    Chunking and compression choices:

    - Large 4D arrays (offset_corr, common_modes, signals) are written with
      one frame per chunk and ``lzf`` compression. Frame-sized chunks match
      the dominant access pattern (frame-wise reads of full pixel rows), and
      lzf has near-zero CPU overhead, which matters for multi-GB arrays.
      Frames are copied one at a time so the archive never holds more than a
      single frame in memory.
    - Small 2D/3D results (median, slopes, msd, signals_mean, pixelwise) are
      written contiguously with ``gzip``: they are tiny, so gzip's better
      compression ratio costs nothing and contiguous layout keeps readers
      simple.

    Parameters
    ----------
    paths : AnalysisPaths
        Locations of the materialized Zarr outputs.
    h5_path : Path
        Destination HDF5 archive path (AnalysisSettings.h5_archive).
    """
    # Tiny results: read fully, write contiguously.
    small_outputs: dict[str, Path] = {
        "median": paths.median,
        "slopes": paths.slopes,
        "msd": paths.msd,
        "signals_mean": paths.signals_mean,
        "pixelwise": paths.pixelwise,
    }
    # Large 4D results: stream one frame at a time.
    large_outputs: dict[str, Path] = {
        "offset_corr": paths.offset_corr,
        "common_modes": paths.common_modes,
        "signals": paths.signals,
    }

    with h5py.File(h5_path, "w") as f:
        for name, source in small_outputs.items():
            data = np.asarray(zarr.open_array(str(source), mode="r"))
            f.create_dataset(name, data=data, compression="gzip")

        for name, source in large_outputs.items():
            arr = zarr.open_array(str(source), mode="r")
            dataset = f.create_dataset(
                name,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=(1, *arr.shape[1:]),  # one frame per chunk
                compression="lzf",
            )
            for frame_idx in range(arr.shape[0]):
                dataset[frame_idx] = np.asarray(arr[frame_idx])
