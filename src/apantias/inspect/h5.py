# h5py has no type stubs, so strict-mode unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""Human-readable inspection and printing for HDF5 archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt


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
    ``apantias.io.hdf5.write_analysis_outputs_to_h5``, but works for any HDF5
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
    """Print summary statistics for a dataset.

    For datasets larger than *sample_max_elements*, a random sample is drawn
    along the first axis so the statistics remain cheap to compute while still
    being representative.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{dataset_name}' is not a dataset in {h5_path}")

        data: npt.NDArray[Any]
        total_elements = int(np.prod(ds.shape))
        if total_elements <= sample_max_elements:
            data = np.asarray(ds[()])
        else:
            # For large arrays sample along the first axis; if even that is too
            # large, fall back to a flat random sample of the requested size.
            n_first = ds.shape[0]
            sample_first = max(1, sample_max_elements // int(np.prod(ds.shape[1:])))
            sample_first = min(sample_first, n_first)
            indices = np.linspace(0, n_first - 1, num=sample_first, dtype=int)
            data = np.asarray(ds[indices])
            if data.size > sample_max_elements:
                flat = data.ravel()
                rng = np.random.default_rng(0)
                flat_indices = rng.choice(len(flat), size=sample_max_elements, replace=False)
                data = flat[flat_indices]
            print(f"  (sampled {data.size} of {total_elements} elements)")

        finite = data[np.isfinite(data)]
        print(f"Stats for '{dataset_name}' in {h5_path}:")
        print(f"  min:    {finite.min() if finite.size else 'n/a'}")
        print(f"  max:    {finite.max() if finite.size else 'n/a'}")
        print(f"  mean:   {finite.mean() if finite.size else 'n/a'}")
        print(f"  std:    {finite.std() if finite.size else 'n/a'}")
        print(f"  finite: {finite.size}/{data.size}")
