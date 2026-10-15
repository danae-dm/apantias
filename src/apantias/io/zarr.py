"""Zarr codecs, path parsing, store inspection and HDF5-to-Zarr conversion."""

# Dask and Zarr expose only partial type information in their shipped stubs; the
# runtime semantics are well understood, so we relax Pyright's strict "unknown"
# checks inside this I/O-heavy module.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import logging
from pathlib import Path
from typing import Any, cast

import dask.array as da
import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import zarr
from dask import delayed  # pyright: ignore
from zarr.codecs import (
    BloscCodec,
    BloscShuffle,
    BytesCodec,
    ShardingCodec,
)

from apantias.settings import AnalysisSettings, FrameSettings

# Helpers live in a package-internal helper module; the underscore signals
# "not part of the public API" rather than "never imported elsewhere".
from ._helpers import _frame_chunk_size, _resolve_range_indices  # pyright: ignore[reportPrivateUsage]
from .hdf5 import _find_h5_dataset, _read_h5_batch  # pyright: ignore[reportPrivateUsage]

_logger = logging.getLogger(__name__)


def _blosc_codecs() -> list[BytesCodec | BloscCodec]:
    """Inner codec pipeline for the raw uint16 detector data.

    LZ4 + bitshuffle is chosen for speed: bitshuffle groups the near-constant
    high bytes of the narrow-range uint16 values together, which already removes
    most of the entropy, and LZ4 is one of the fastest Blosc codecs.
    """
    return [
        BytesCodec(endian="little"),
        BloscCodec(
            cname="lz4",
            clevel=5,
            shuffle=BloscShuffle.bitshuffle,
        ),
    ]


def _sharded_frame_codecs(inner_chunk_shape: tuple[int, ...]) -> list[ShardingCodec]:
    """Codec pipeline for the frame-chunked array using a sharding codec."""
    return [
        ShardingCodec(
            chunk_shape=inner_chunk_shape,
            codecs=_blosc_codecs(),
        )
    ]


def _parse_zarr_path(zarr_path: str | Path) -> tuple[str, str | None, str]:
    """Split a ``/path/to/store.zarr/group/path/dataset`` path.

    Returns the store path, the (possibly ``None``) group path, and the dataset
    name.
    """
    path_str = str(zarr_path)
    if ".zarr" not in path_str:
        raise ValueError(f"zarr_path must contain '.zarr': {path_str}")

    store_end = path_str.find(".zarr") + len(".zarr")
    store_path = path_str[:store_end]
    remainder = path_str[store_end:].lstrip("/")

    if not remainder:
        raise ValueError(f"zarr_path must specify group and dataset: {path_str}")

    parts = remainder.split("/")
    dataset_name = parts[-1]
    group_path = "/".join(parts[:-1]) if len(parts) > 1 else None
    return store_path, group_path, dataset_name


def get_chunk_info(zarr_path: str) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Return the outer and inner chunk shape for a Zarr v3 array."""
    # zarr.open may return Array or Group; this helper only makes sense for Arrays.
    z = cast(zarr.Array[Any], zarr.open(zarr_path, mode="r"))

    # zarr's type stubs expose these attributes dynamically on the metadata object.
    outer: tuple[int, ...] = z.metadata.chunk_grid.chunk_shape  # type: ignore[attr-defined]

    inner_chunk: tuple[int, ...] | None = None
    for codec in z.metadata.codecs:  # type: ignore[attr-defined]
        if hasattr(codec, "chunk_shape"):  # type: ignore[attr-defined]
            inner_chunk = codec.chunk_shape  # type: ignore[attr-defined]
            break

    return outer, inner_chunk


def print_store_info(zarr_path: str) -> None:
    """Print a combined tree and chunk layout summary for a zarr store."""

    def _lines(
        node: zarr.Array[Any] | zarr.Group,
        prefix: str = "",
        last: bool = True,
    ) -> list[str]:
        connector = "└── " if last else "├── "
        name = node.name.split("/")[-1] or zarr_path

        if isinstance(node, zarr.Array):
            meta = node.metadata
            outer: tuple[int, ...] = meta.chunk_grid.chunk_shape  # type: ignore[attr-defined]
            inner: tuple[int, ...] | None = next(
                (c.chunk_shape for c in meta.codecs if hasattr(c, "chunk_shape")),  # type: ignore[attr-defined]
                None,
            )
            size_mb = node.nbytes_stored() / 1024**2
            header = f"{prefix}{connector}{name}  {node.shape}  {node.dtype}  ({size_mb:.1f} MB on disk)"
            child_prefix = prefix + ("    " if last else "│   ")
            if inner:
                detail = [
                    f"{child_prefix}shard : {outer}",
                    f"{child_prefix}chunk : {inner}",
                ]
            else:
                detail = [f"{child_prefix}chunk : {outer}"]
            return [header] + detail

        # Group
        header = f"{prefix}{connector}{name}/"
        child_prefix = prefix + ("    " if last else "│   ")
        result = [header]
        members = list(node.members())
        for i, (_, child) in enumerate(members):
            result.extend(_lines(child, child_prefix, last=(i == len(members) - 1)))
        return result

    # zarr.open may return Array or Group; both are handled below.
    root: zarr.Array[Any] | zarr.Group = zarr.open(zarr_path, mode="r")  # type: ignore[assignment]
    lines = [zarr_path]
    if isinstance(root, zarr.Group):
        members = list(root.members())
        for i, (_, child) in enumerate(members):
            lines.extend(_lines(child, "", last=(i == len(members) - 1)))
    else:
        lines.extend(_lines(root, "", last=True))
    print("\n".join(lines))


def h5_to_zarr(
    h5_path: str | Path,
    zarr_path: str | Path,
    frame: FrameSettings,
    analysis: AnalysisSettings,
    dataset_name: str | None = None,
) -> Path:
    """Write the data of an HDF5 dataset to a Zarr v3 store."""
    h5_path = Path(h5_path)
    zarr_path = Path(zarr_path)

    store_path, group_path, output_dataset_name = _parse_zarr_path(zarr_path)

    # Read the layout metadata and resolve the source dataset.
    with h5py.File(h5_path, "r") as f:
        if dataset_name is None:
            dataset_name = _find_h5_dataset(f)
        # h5py has no type stubs; cast to Dataset so shape/dtype are treated as known.
        ds = cast(h5py.Dataset, f[dataset_name])
        shape: tuple[int, ...] = ds.shape  # type: ignore[attr-defined]
        raw_nreps: int = shape[2]
        n_frames: int = shape[0]
        if shape[1] != frame.cols or shape[3] != frame.rows:
            raise ValueError(
                f"Unexpected frame layout {shape} in {h5_path}:{dataset_name}; "
                f"expected (n_frames, {frame.cols}, nreps, {frame.rows})"
            )

    # Resolve nreps slice from analysis settings
    nreps_indices = _resolve_range_indices(analysis.nreps_range, raw_nreps)
    eval_nreps = len(nreps_indices)

    chunk_size = _frame_chunk_size(eval_nreps, frame.cols, frame.rows, frame.target_chunk_bytes)

    # Resolve frame indices from analysis settings
    frame_indices = _resolve_range_indices(analysis.frames_range, n_frames)
    n_output_frames = len(frame_indices)

    array_shape = (n_output_frames, frame.cols, eval_nreps, frame.rows)
    chunk_shape = (chunk_size, frame.cols, eval_nreps, frame.rows)
    inner_chunk_shape = (chunk_size, 1, eval_nreps, frame.rows)

    full_array_path = (
        f"{store_path}/{group_path}/{output_dataset_name}" if group_path else f"{store_path}/{output_dataset_name}"
    )

    target = zarr.open_array(
        full_array_path,
        mode="w",
        shape=array_shape,
        chunks=chunk_shape,
        dtype="uint16",
        zarr_format=3,
        codecs=_sharded_frame_codecs(inner_chunk_shape),
    )

    n_batches = (n_output_frames + chunk_size - 1) // chunk_size
    _logger.info(
        "Building Dask array of %d frames (%d nreps) in %d chunks (chunk_size=%d frames)...",
        n_output_frames,
        eval_nreps,
        n_batches,
        chunk_size,
    )

    blocks: list[da.Array] = []
    for batch_start in range(0, n_output_frames, chunk_size):
        batch_end = min(batch_start + chunk_size, n_output_frames)

        # Translate output-frame offsets to actual HDF5 frame indices
        h5_start = frame_indices[batch_start]
        h5_end = frame_indices[batch_end] if batch_end < n_output_frames else frame_indices[-1] + 1

        block = da.from_delayed(  # type: ignore[reportUnknownMemberType]
            delayed(_read_h5_batch)(  # type: ignore[reportUnknownArgumentType]
                h5_path,
                dataset_name,
                h5_start,
                h5_end,
                frame_indices=frame_indices,
                nreps_indices=nreps_indices,
            ),
            shape=(batch_end - batch_start, frame.cols, eval_nreps, frame.rows),
            dtype=np.uint16,
        )
        blocks.append(block)

    data: da.Array = da.concatenate(blocks, axis=0)  # type: ignore[reportUnknownMemberType]
    da.store(data, target, lock=False)  # type: ignore[reportUnknownMemberType]

    _logger.info("Successfully wrote %d frames to %s", n_output_frames, zarr_path)
    return zarr_path
