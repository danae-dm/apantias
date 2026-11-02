# h5py has no type stubs, so strict-mode unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""Human-readable inspection and printing for HDF5 archives."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np


def _format_bytes(nbytes: int) -> str:
    """Return a human-readable byte count."""
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def _walk_datasets(
    group: h5py.Group,
    depth: int,
    lines: list[str],
    prefix: str = "",
) -> None:
    """Recursively collect formatted lines for every group and dataset."""
    for key in group:
        obj = group[key]
        key_str = str(key)
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            size = _format_bytes(obj.nbytes)
            lines.append(
                f"{prefix}{indent}{key_str}  shape={obj.shape}  dtype={obj.dtype}  "
                f"size={size}  chunks={obj.chunks}  compression={obj.compression}"
            )
        elif isinstance(obj, h5py.Group):
            lines.append(f"{prefix}{indent}{key_str}/")
            _walk_datasets(obj, depth + 1, lines, prefix)


def print_h5_contents(h5_path: Path | str) -> None:
    """Print a tree of all groups and datasets in an HDF5 file."""
    h5_path = Path(h5_path)
    file_size = _format_bytes(h5_path.stat().st_size)
    print(f"{h5_path} ({file_size})")

    with h5py.File(h5_path, "r") as f:
        lines: list[str] = []
        _walk_datasets(f, 0, lines)
        print("\n".join(lines))


def print_archive_summary(h5_path: Path | str) -> None:
    """Print a compact table of all top-level datasets in an archive file.

    This is tailored to the standard analysis archive produced by
    ``apantias.storage.hdf5.write_analysis_outputs_to_h5``, but works for any HDF5
    file with flat top-level datasets.
    """
    h5_path = Path(h5_path)
    file_size = _format_bytes(h5_path.stat().st_size)
    print(f"Archive: {h5_path} ({file_size})")

    with h5py.File(h5_path, "r") as f:
        datasets: list[tuple[str, h5py.Dataset]] = []
        for key in f:
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                datasets.append((str(key), obj))

        if not datasets:
            print("  (no top-level datasets)")
            return

        name_width = max(len(name) for name, _ in datasets)
        shape_width = max(len(str(ds.shape)) for _, ds in datasets)
        print(f"{'Dataset':<{name_width}}  {'Shape':<{shape_width}}  Dtype      Size")
        for name, ds in datasets:
            print(
                f"{name:<{name_width}}  {str(ds.shape):<{shape_width}}  {str(ds.dtype):<9}  {_format_bytes(ds.nbytes)}"
            )


def print_dataset_info(h5_path: Path | str, dataset_name: str) -> None:
    """Print detailed metadata for a single dataset."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{dataset_name}' is not a dataset in {h5_path}")

        print(f"Dataset: {dataset_name} in {h5_path}")
        print(f"  shape:       {ds.shape}")
        print(f"  dtype:       {ds.dtype}")
        print(f"  size:        {_format_bytes(ds.nbytes)}")
        print(f"  chunks:      {ds.chunks}")
        print(f"  compression: {ds.compression}")
        print(f"  attrs:       {dict(ds.attrs)}")


def print_dataset_stats(
    h5_path: Path | str,
    dataset_name: str,
    *,
    sample_max_elements: int = 1_000_000,
) -> None:
    """Print basic statistics for a single dataset.

    For datasets larger than *sample_max_elements*, a regularly-spaced subset
    along axis 0 is read to keep the operation fast on chunked data. The
    output is then marked as "sampled".
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{dataset_name}' is not a dataset in {h5_path}")

        total_elements = int(np.prod(ds.shape))
        if total_elements <= sample_max_elements or ds.ndim == 0:
            data = np.asarray(ds[()])
            sampled = False
        else:
            # Chunked data is along axis 0; stride that axis to stay efficient.
            stride = max(1, int(total_elements // sample_max_elements))
            idx = (slice(None, None, stride),) + (slice(None),) * (ds.ndim - 1)
            data = np.asarray(ds[idx])
            sampled = True

        finite = np.isfinite(data)
        finite_data = data[finite]
        prefix = "sampled " if sampled else ""
        print(f"Stats for {dataset_name} in {h5_path}:")
        print(f"  shape:       {ds.shape}")
        print(f"  dtype:       {ds.dtype}")
        print(f"  {prefix}min:        {finite_data.min() if finite_data.size else 'n/a'}")
        print(f"  {prefix}max:        {finite_data.max() if finite_data.size else 'n/a'}")
        print(f"  {prefix}mean:       {finite_data.mean() if finite_data.size else 'n/a'}")
        print(f"  {prefix}std:        {finite_data.std() if finite_data.size else 'n/a'}")
        print(f"  finite:      {finite.sum()} / {data.size}")


def read_dataset_slice(
    h5_path: Path | str,
    dataset_name: str,
    slice_spec: slice | Sequence[slice] | np.ndarray | None = None,
) -> np.ndarray:
    """Return a slice of a dataset as a NumPy array.

    Parameters
    ----------
    h5_path:
        Path to the HDF5 file.
    dataset_name:
        Name of the dataset (supports ``"group/dataset"`` syntax).
    slice_spec:
        What to read. Accepts:

        - ``None`` – read the entire dataset.
        - A single ``slice`` – e.g. ``slice(0, 100)`` to get the first 100
          entries along axis 0.
        - A tuple of ``slice`` objects – e.g. ``(slice(0, 100), slice(None),
          slice(None))`` or simply ``slice(0, 100)`` which is expanded to
          ``slice(None)`` for remaining axes.
        - A 1-D integer ``np.ndarray`` – interpretted as row indices along
          axis 0 (advanced indexing).

    Returns
    -------
    ``np.ndarray``
        The requested slice of the data.

    Raises
    ------
    KeyError
        If *dataset_name* does not exist.
    ValueError
        If *slice_spec* is incompatible with the dataset shape.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{dataset_name}' is not a dataset in {h5_path}")

        if slice_spec is None:
            index = ()
        elif isinstance(slice_spec, slice):
            # Expand a single slice to "full" for remaining axes
            # e.g. slice(10, 20) -> (10:20, ..., ...)
            index = (slice_spec,) + (slice(None),) * (len(ds.shape) - 1)
        elif isinstance(slice_spec, tuple):
            # Validate that the tuple matches the rank
            # (h5py would raise, but give a clearer message)
            if len(slice_spec) != len(ds.shape):
                raise ValueError(
                    f"slice tuple length {len(slice_spec)} does not match "
                    f"dataset rank {len(ds.shape)} for '{dataset_name}'"
                )
            index = slice_spec
        elif (
            # duck-typing for numpy array
            hasattr(slice_spec, "ndim")
            and slice_spec.ndim == 1  # type: ignore[union-attr]
            and np.issubdtype(slice_spec.dtype, np.integer)  # type: ignore[union-attr]
        ):
            # Integer index array -> advanced indexing on axis 0
            index = (slice_spec,) + (slice(None),) * (len(ds.shape) - 1)
        else:
            raise TypeError(
                f"Unsupported slice_spec type {type(slice_spec).__name__}; "
                f"expected slice, tuple[slice, ...], or 1-D int ndarray"
            )

        return np.asarray(ds[index])
