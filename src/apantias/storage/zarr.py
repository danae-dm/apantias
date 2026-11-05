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
import numpy.typing as npt
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


def _blosc_codecs(dtype: np.dtype[Any] | None = None) -> list[BytesCodec | BloscCodec]:
    """Inner codec pipeline for detector data.

    LZ4 + bitshuffle is chosen for speed: bitshuffle groups the near-constant
    high bytes of the narrow-range values together, which already removes
    most of the entropy, and LZ4 is one of the fastest Blosc codecs.

    Parameters
    ----------
    dtype:
        Data type of the array being compressed. The codec pipeline is
        suitable for both uint16 and 4-byte integer/float dtypes; compressed
        size may vary depending on dtype. Defaults to uint16 if not specified.
    """
    if dtype is None:
        dtype = np.dtype("uint16")
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
    *,
    store_batch_size: int | None = None,
) -> Path:
    """Write the data of an HDF5 dataset to a Zarr v3 store.

    Parameters
    ----------
    store_batch_size:
        If given, ``da.store`` is called in waves of at most this many shards
        instead of submitting the whole array in one graph. This bounds the
        number of in-flight tasks and the transient memory per worker.
    """
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
        if shape[1] != frame.rows or shape[3] != frame.cols:
            raise ValueError(
                f"Unexpected frame layout {shape} in {h5_path}:{dataset_name}; "
                f"expected (n_frames, {frame.rows}, nreps, {frame.cols})"
            )

    # Resolve nreps slice from analysis settings
    nreps_indices = _resolve_range_indices(analysis.nreps_range, raw_nreps)
    eval_nreps = len(nreps_indices)

    chunk_size = _frame_chunk_size(eval_nreps, frame.rows, frame.cols, frame.target_chunk_bytes)

    # Resolve frame indices from analysis settings
    frame_indices = _resolve_range_indices(analysis.frames_range, n_frames)
    n_output_frames = len(frame_indices)

    array_shape = (n_output_frames, frame.rows, eval_nreps, frame.cols)
    chunk_shape = (chunk_size, frame.rows, eval_nreps, frame.cols)
    inner_chunk_shape = (chunk_size, 1, eval_nreps, frame.cols)

    # Determine storage dtype based on raw_sign_flip
    raw_sign_flip = analysis.raw_sign_flip
    _logger.info("h5_to_zarr: raw_sign_flip=%s", raw_sign_flip)

    # Store dtype info for logging later
    storage_dtype: np.dtype[Any]
    storage_dtype_name: str

    if raw_sign_flip:
        # When sign flip is active, store as float32.
        # Float32 is chosen because:
        # 1. It handles all uint16 values (0-65535) without overflow when negated
        # 2. Simple and predictable behavior
        # 3. Good compression with Blosc/bitshuffle
        storage_dtype = np.dtype("float32")
        storage_dtype_name = "float32"
        _logger.info(
            "h5_to_zarr: using float32 storage for sign-flipped data. "
            "Values will be cast from uint16 -> int32 -> negate -> float32."
        )
    else:
        # Original behavior: uint16, no sign flip
        storage_dtype = np.dtype("uint16")
        storage_dtype_name = "uint16"
        bytes_per_frame = frame.rows * eval_nreps * frame.cols * 2
        total_output_mb = np.prod(array_shape, dtype=np.int64) * 2 / 1e6

        _logger.info(
            "h5_to_zarr: input shape %s, output shape %s, chunk shape %s, bytes per frame %d, "
            "total output %.2f MB, storage dtype=%s (no sign flip).",
            shape,
            array_shape,
            chunk_shape,
            bytes_per_frame,
            total_output_mb,
            storage_dtype_name,
        )

    full_array_path = (
        f"{store_path}/{group_path}/{output_dataset_name}" if group_path else f"{store_path}/{output_dataset_name}"
    )

    target = zarr.open_array(
        full_array_path,
        mode="w",
        shape=array_shape,
        chunks=chunk_shape,
        dtype=storage_dtype,
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

        # Create a delayed function that reads and optionally negates the data
        def _read_and_transform(
            h5_path: Path,
            dataset_name: str,
            batch_start_idx: int,
            batch_end_idx: int,
            frame_indices: list[int],
            nreps_indices: list[int],
            raw_sign_flip: bool,
        ) -> npt.NDArray[Any]:
            """Read batch from HDF5 and apply sign flip if configured."""
            batch = _read_h5_batch(h5_path, dataset_name, batch_start_idx, batch_end_idx, frame_indices, nreps_indices)

            if raw_sign_flip:
                # Cast to int32 first, then negate to avoid uint16->int16 overflow
                return batch.astype(np.int32) * -1
            else:
                return batch

        block = da.from_delayed(  # type: ignore[reportUnknownMemberType]
            delayed(_read_and_transform)(  # type: ignore[reportUnknownArgumentType]
                h5_path,
                dataset_name,
                h5_start,
                h5_end,
                frame_indices=frame_indices,
                nreps_indices=nreps_indices,
                raw_sign_flip=raw_sign_flip,
            ),
            shape=(batch_end - batch_start, frame.rows, eval_nreps, frame.cols),
            dtype=storage_dtype,
        )
        blocks.append(block)
        block_num = len(blocks)
        if block_num % 25 == 0 or block_num == n_batches:
            _logger.info("Created %d/%d dask blocks (graph construction only).", block_num, n_batches)

    _logger.info("Finished creating blocks; about to concatenate.")
    data: da.Array = da.concatenate(blocks, axis=0)  # type: ignore[reportUnknownMemberType]
    _logger.info(
        "Ready to da.store: %d blocks, shape=%s, chunks=%s, nbytes=%.2f MB.",
        len(blocks),
        data.shape,
        data.chunksize,  # type: ignore[attr-defined]
        data.nbytes / 1e6,  # type: ignore[attr-defined]
    )

    if store_batch_size is None or len(blocks) <= store_batch_size:
        da.store(data, target, lock=False)  # type: ignore[reportUnknownMemberType]
    else:
        _logger.info(
            "Storing in batches of %d blocks (%d batches total).",
            store_batch_size,
            (len(blocks) + store_batch_size - 1) // store_batch_size,
        )
        n_full_batches = len(blocks) // store_batch_size
        for batch_idx in range(n_full_batches + (1 if len(blocks) % store_batch_size else 0)):
            batch_start = batch_idx * store_batch_size
            batch_end = min(batch_start + store_batch_size, len(blocks))
            batch_blocks = blocks[batch_start:batch_end]
            batch_data: da.Array = da.concatenate(batch_blocks, axis=0)  # type: ignore[reportUnknownMemberType]

            out_start = batch_start * chunk_size
            out_end = min(out_start + len(batch_blocks) * chunk_size, n_output_frames)
            region = (slice(out_start, out_end), slice(None), slice(None), slice(None))

            _logger.info(
                "Storing batch %d/%d: blocks %d-%d -> output frames %d-%d.",
                batch_idx + 1,
                (len(blocks) + store_batch_size - 1) // store_batch_size,
                batch_start,
                batch_end - 1,
                out_start,
                out_end,
            )
            da.store(
                batch_data,
                target,
                lock=False,
                regions=[region],
            )  # type: ignore[reportUnknownMemberType]

    _logger.info("Successfully wrote %d frames to %s with dtype=%s", n_output_frames, zarr_path, storage_dtype_name)
    return zarr_path
