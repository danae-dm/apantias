import logging
import os
from pathlib import Path
from typing import Any, cast

import dask.array as da
import h5py
import hdf5plugin
import numpy as np
import zarr
from dask import delayed  # pyright: ignore
from dask.base import compute
from zarr.codecs import (
    BloscCodec,
    BloscShuffle,
    BytesCodec,
    ShardingCodec,
)

_logger = logging.getLogger(__name__)

_COLUMN_SIZE = 64
_ROW_SIZE = 64
_KEY_INTS = 3
_RAW_ROW_SIZE = _ROW_SIZE + _KEY_INTS  # 67 uint16 values per raw row
_TARGET_CHUNK_BYTES = 100 * 1024 * 1024  # 200 MB
_NREPS_EVAL = slice(3, None, 1)


def get_node_name() -> str:
    """Get the name of the compute node the process is running on.

    Uses the SLURMD_NODENAME environment variable set by SLURM,
    or falls back to the hostname if not running on a SLURM cluster.

    Returns
    -------
    str
        The name of the node.
    """
    node_name = os.environ.get("SLURMD_NODENAME")
    if node_name is None:
        import socket

        node_name = socket.gethostname()
    return node_name


def _blosc_codecs() -> list:
    """Inner codec pipeline for the raw uint16 detector data.

    LZ4 + bitshuffle is chosen for speed: bitshuffle groups the near-constant
    high bytes of the narrow-range uint16 values together, which already removes
    most of the entropy, and LZ4 is one of the fastest Blosc codecs (several
    times faster to compress than Zstd-9) with very fast decompression. The
    resulting ratio is moderate but more than enough here, and copying to the
    fast scratch drive becomes compression-bound rather than IO-bound far less
    often.
    """
    return [
        BytesCodec(endian="little"),
        BloscCodec(
            cname="lz4",
            clevel=5,
            shuffle=BloscShuffle.bitshuffle,
        ),
    ]


def _sharded_frame_codecs(inner_chunk_shape: tuple[int, ...]) -> list:
    """Codec pipeline for the frame-chunked array using a sharding codec.

    The Zarr chunk (= the write/transfer unit, ~100 MB) becomes a *shard* that
    is internally subdivided into ``inner_chunk_shape`` inner chunks, each
    compressed separately with :func:`_blosc_codecs`.

    Making the inner chunk one column wide ``(chunk_size, 1, nreps, n_row)`` is
    what accelerates the downstream pixel rechunk: reading a single column then
    becomes a *partial shard read* that decompresses only that column's inner
    chunks instead of the whole shard. Across all 64 column passes every inner
    chunk is decompressed exactly once (1x total) rather than once per pass
    (64x), with no increase in per-task memory.
    """
    return [
        ShardingCodec(
            chunk_shape=inner_chunk_shape,
            codecs=_blosc_codecs(),
        )
    ]


def _read_frame_batch(
    bin_file: Path,
    offset: int,
    frame_start_indices: np.ndarray,
    frame_end_indices: np.ndarray,
    batch_start: int,
    batch_end: int,
    nreps: int,
    nreps_slice: slice,
) -> np.ndarray:
    """Extract one batch of frames from the binary file as a NumPy array.

    This is the per-chunk worker for the Dask array. It memory-maps the file
    (only the pages for *this* batch are paged in by the OS), gathers the frame
    rows for ``batch_start:batch_end`` and returns a
    ``(batch, COLUMN_SIZE, eval_nreps, ROW_SIZE)`` array. The only RAM it holds is
    that single decoded batch, so total memory is bounded by the number of
    Dask tasks running concurrently, not by the file size.
    """
    raw_uint16 = np.memmap(bin_file, dtype="uint16", mode="r", offset=offset)
    n_complete_rows = len(raw_uint16) // _RAW_ROW_SIZE
    raw_data = raw_uint16[: n_complete_rows * _RAW_ROW_SIZE].reshape(-1, _RAW_ROW_SIZE)

    batch = np.stack([
        raw_data[frame_start_indices[i] + 1 : frame_end_indices[i] + 1, :_ROW_SIZE]
        for i in range(batch_start, batch_end)
    ])
    # Reshape to full nreps, apply the slice, then return
    batch_full = batch.reshape(-1, _COLUMN_SIZE, nreps, _ROW_SIZE)
    return batch_full[:, :, nreps_slice, :]


def _frame_chunk_size(nreps: int) -> int:
    """Number of frames per write/compression chunk for the given ``nreps``.

    Bounded so each chunk is at most ``_TARGET_CHUNK_BYTES`` (~100 MB), then
    rounded down to the nearest multiple of 10 (or kept as 1 if tiny), so the
    same write-unit budget is used for the HDF5 and Zarr pipelines.
    """
    bytes_per_frame = _COLUMN_SIZE * nreps * _ROW_SIZE * 2
    chunk_size = max(1, _TARGET_CHUNK_BYTES // bytes_per_frame)
    if chunk_size >= 10:
        chunk_size = (chunk_size // 10) * 10
    return chunk_size


def _parse_bin_frames(
    bin_file: Path,
    offset: int,
    nreps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Locate the valid frame boundaries in a binary file.

    Memory-maps the raw file (only the pages actually touched are loaded by the
    OS), finds every frame-key row (sentinel 65535 at column ``_COLUMN_SIZE``)
    and returns the start/end row indices of consecutive key pairs whose
    interiors span exactly ``_COLUMN_SIZE * nreps`` rows, plus the frame count.
    """
    rows_per_frame = _COLUMN_SIZE * nreps
    raw_uint16 = np.memmap(bin_file, dtype="uint16", mode="r", offset=offset)
    n_complete_rows = len(raw_uint16) // _RAW_ROW_SIZE
    raw_data = raw_uint16[: n_complete_rows * _RAW_ROW_SIZE].reshape(-1, _RAW_ROW_SIZE)

    # Locate all frame-key rows (sentinel 65535 at column _COLUMN_SIZE).
    frame_key_positions = np.where(raw_data[:, _COLUMN_SIZE] == 65535)[0]
    if len(frame_key_positions) < 2:
        raise ValueError(f"No valid frames found in {bin_file}")
    # A valid frame has exactly rows_per_frame rows between two consecutive keys.
    starts = frame_key_positions[:-1]
    ends = frame_key_positions[1:]
    valid_mask = (ends - starts) == rows_per_frame
    frame_start_indices = starts[valid_mask]
    frame_end_indices = ends[valid_mask]
    n_frames = len(frame_start_indices)

    if n_frames == 0:
        raise ValueError(f"No valid frames found in {bin_file}")
    return frame_start_indices, frame_end_indices, n_frames


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


def bin_to_zarr(
    bin_file: str | Path,
    zarr_path: str | Path,
    nreps: int,
    offset: int = 8,
) -> Path:
    """
    Reads frames from a binary file and writes them to a Zarr v3 store.

    The work is expressed as a **Dask array** so progress is visible in the
    Dask dashboard and execution is distributed across the active cluster. The
    array is built from one lazy block per Zarr chunk; each block is produced by
    :func:`_read_frame_batch`, which memory-maps the file and decodes only that
    batch. ``to_zarr`` then streams the blocks straight into the compressed
    store.

    Memory footprint is bounded by *concurrent* tasks, not file size: at any
    moment Dask holds roughly ``n_running_tasks`` decoded batches in RAM, each
    about ``chunk_size * COLUMN_SIZE * nreps * ROW_SIZE * 2`` bytes (the
    ``_TARGET_CHUNK_BYTES`` budget, ~100 MB). With one thread per worker and N
    workers, peak RAM ≈ ``N * ~100 MB`` regardless of how many frames the file
    contains.

    The binary file is memory-mapped to handle files larger than available RAM.
    The output array is stored as "raw_data" in the zarr group and has shape
    ``(n_frames, COLUMN_SIZE, nreps, ROW_SIZE)``.

    Compression uses Blosc/Zstd with bitshuffle, which is highly effective for
    uint16 values clustered in a narrow range (e.g. ~47000 ± 100): bitshuffle
    groups nearly-identical high-bytes together, giving Zstd very high entropy
    reduction before final encoding.

    Args:
        bin_file: Path to the source .bin file containing uint16 values.
        zarr_path: Path where the Zarr v3 store will be created.
        nreps: Number of repetitions per frame column.
        offset: Byte offset into the binary file to start reading from.

    Returns:
        Path to the zarr store.
    """
    bin_file = Path(bin_file)
    zarr_path = Path(zarr_path)

    # Locate all valid frames; see _parse_bin_frames.
    frame_start_indices, frame_end_indices, n_frames = _parse_bin_frames(bin_file, offset, nreps)
    _logger.info("Found %d valid frames in %s", n_frames, bin_file)

    # ------------------------------------------------------------------
    # Parse zarr_path to extract store, group, and dataset name.
    # Expected format: /path/to/store.zarr/group/path/dataset_name
    # ------------------------------------------------------------------
    store_path, group_path, dataset_name = _parse_zarr_path(zarr_path)

    # Calculate how many reps remain after slicing, then derive the chunk size
    # from the ~100 MB write-unit budget.
    eval_nreps = len(np.empty(nreps)[_NREPS_EVAL])
    chunk_size = _frame_chunk_size(eval_nreps)

    # ------------------------------------------------------------------
    # Zarr v3 array with a sharding codec.
    #   shard (= write/transfer unit, ~100 MB): (chunk_size, 64, nreps, 64)
    #   inner chunk (= compression unit): (chunk_size, 1, nreps, 64)
    # The per-column inner chunk lets the downstream pixel rechunk read one
    # column via a partial shard read, decompressing each inner chunk only once.
    # See _sharded_frame_codecs() for details.
    # ------------------------------------------------------------------
    array_shape = (n_frames, _COLUMN_SIZE, eval_nreps, _ROW_SIZE)
    chunk_shape = (chunk_size, _COLUMN_SIZE, eval_nreps, _ROW_SIZE)
    inner_chunk_shape = (chunk_size, 1, eval_nreps, _ROW_SIZE)

    # Construct full zarr array path
    full_array_path = f"{store_path}/{group_path}/{dataset_name}" if group_path else f"{store_path}/{dataset_name}"

    # We create the empty array structure here so we control the v3 codec
    # pipeline. Dask's store writes into it.
    target = zarr.open_array(
        full_array_path,
        mode="w",
        shape=array_shape,
        chunks=chunk_shape,
        dtype="uint16",
        zarr_format=3,
        codecs=_sharded_frame_codecs(inner_chunk_shape),
    )

    # ------------------------------------------------------------------
    # Build a lazy Dask array: one block per Zarr chunk along the frame axis.
    #
    # Each block is produced on a worker by _read_frame_batch, which decodes
    # only that batch from the memmap. Dask schedules these across the cluster
    # (visible in the dashboard) and, crucially, only keeps the blocks of
    # *currently running* tasks in memory. Peak RAM therefore scales with the
    # number of concurrent tasks, NOT with n_frames -> safe for 800 nreps and
    # thousands of frames.
    # ------------------------------------------------------------------
    n_batches = (n_frames + chunk_size - 1) // chunk_size
    _logger.info(
        "Building Dask array of %d frames in %d chunks (chunk_size=%d frames)...",
        n_frames,
        n_batches,
        chunk_size,
    )

    blocks = []
    for batch_start in range(0, n_frames, chunk_size):
        batch_end = min(batch_start + chunk_size, n_frames)
        block = da.from_delayed(
            delayed(_read_frame_batch)(
                bin_file,
                offset,
                frame_start_indices,
                frame_end_indices,
                batch_start,
                batch_end,
                nreps,
                _NREPS_EVAL,
            ),
            shape=(batch_end - batch_start, _COLUMN_SIZE, eval_nreps, _ROW_SIZE),
            dtype=np.uint16,
        )
        blocks.append(block)

    data = da.concatenate(blocks, axis=0)

    # Stream the blocks into the pre-created compressed Zarr array. Writing into
    # the existing array object preserves our Blosc/bitshuffle codec pipeline.
    # Dask shows task progress in the dashboard while writing.
    da.store(data, target, lock=False)

    _logger.info("Successfully wrote %d frames to %s", n_frames, zarr_path)
    return zarr_path


def bin_to_h5(
    bin_path: str | Path,
    nreps: int,
    h5_path: str | Path,
    dataset_name: str,
    offset: int = 8,
) -> Path:
    """Write the frames of a binary file to a compressed HDF5 dataset.

    The no-Dask counterpart of :func:`bin_to_zarr`: it parses the same frame
    boundaries (via :func:`_parse_bin_frames`) but keeps **all** repetitions and
    writes the full-resolution data as a uint16 dataset of shape
    ``(n_frames, COLUMN_SIZE, nreps, ROW_SIZE)``. Dropping the first three
    repetitions stays a property of the zarr target and is applied by
    :func:`h5_to_zarr`.

    The dataset is compressed with Blosc (zstd-level-9 + bitshuffle) via
    ``hdf5plugin``, mirroring the bitshuffle codec of the zarr store: the
    narrow-range uint16 values have near-constant high bytes, so bitshuffle
    groups them into zero-heavy bitplanes that Zstd then encodes almost for
    free. uint16 is the natural storage dtype — the values (~47000 ± 100)
    exceed int16, and any wider dtype would only compress fewer values per byte.

    Frames are written in batches of ``chunk_size`` (see
    :func:`_frame_chunk_size`) so memory stays bounded at roughly
    ``_TARGET_CHUNK_BYTES`` regardless of the number of frames. The metadata
    needed by :func:`h5_to_zarr` (``nreps``, frame count, layout) is stored as
    attributes on the dataset.

    Args:
        bin_path: Path to the source .bin file containing uint16 values.
        nreps: Number of repetitions per frame column.
        h5_path: Path where the HDF5 file will be created (its parent directory
            must exist).
        dataset_name: Name of the dataset within the HDF5 file.
        offset: Byte offset into the binary file to start reading from.

    Returns:
        Path to the created HDF5 file.
    """
    bin_path = Path(bin_path)
    h5_path = Path(h5_path)

    frame_start_indices, frame_end_indices, n_frames = _parse_bin_frames(bin_path, offset, nreps)
    _logger.info("Found %d valid frames in %s", n_frames, bin_path)

    chunk_size = _frame_chunk_size(nreps)
    filters = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=1)
    shape = (n_frames, _COLUMN_SIZE, nreps, _ROW_SIZE)
    chunk_shape = (chunk_size, _COLUMN_SIZE, nreps, _ROW_SIZE)

    # Keep all reps (no slice)
    nreps_slice = slice(None)

    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset(dataset_name, shape=shape, dtype="uint16", chunks=chunk_shape, **filters)
        ds.attrs["nreps"] = nreps
        ds.attrs["offset"] = offset
        ds.attrs["n_frames"] = n_frames
        ds.attrs["column_size"] = _COLUMN_SIZE
        ds.attrs["row_size"] = _ROW_SIZE
        ds.attrs["raw_row_size"] = _RAW_ROW_SIZE

        # Build delayed tasks — one per batch
        delayed_tasks = []
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
                )
            )

        # Execute in parallel (threads are fine here — GIL released by
        # numpy/hdf5plugin under the hood)
        compute(*delayed_tasks, scheduler="threads")

    return h5_path


def _write_h5_batch(ds, start, end, bin_path, offset, frame_start_indices, frame_end_indices, nreps, nreps_slice):
    """Read one batch from binary and write it to a pre-opened h5py dataset."""
    batch = _read_frame_batch(bin_path, offset, frame_start_indices, frame_end_indices, start, end, nreps, nreps_slice)
    ds[start:end] = batch


def _read_h5_batch(
    h5_path: Path,
    dataset_name: str,
    batch_start: int,
    batch_end: int,
) -> np.ndarray:
    """Read one batch of frames from an HDF5 dataset as a NumPy array.

    This is the per-chunk worker for the Dask array in :func:`h5_to_zarr`. It
    opens the HDF5 file *inside* the task (h5py file handles cannot cross
    process boundaries) and returns ``(batch, COLUMN_SIZE, eval_nreps, ROW_SIZE)``
    by applying the evaluation-rep slice. The only RAM it holds is that single
    decoded batch.
    """
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"{dataset_name} is not a dataset in {h5_path}")
        batch = ds[batch_start:batch_end]

    return batch[:, :, _NREPS_EVAL, :]


def _find_h5_dataset(f: h5py.File) -> str:
    """Return the single dataset path in an HDF5 file.

    Used by :func:`h5_to_zarr` when no ``dataset_name`` is given; files written
    by :func:`bin_to_h5` contain exactly one dataset.
    """
    datasets: list[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    f.visititems(visit)
    if len(datasets) != 1:
        raise ValueError(f"Expected exactly one dataset in {f.filename}, found {len(datasets)}: {datasets}")
    return datasets[0]


def h5_to_zarr(
    h5_path: str | Path,
    zarr_path: str | Path,
    dataset_name: str | None = None,
) -> Path:
    """Write the data of an HDF5 dataset to a Zarr v3 store.

    The Dask-based counterpart of :func:`bin_to_zarr` for data that already went
    through :func:`bin_to_h5`. It produces the same store that ``bin_to_zarr``
    would: full-resolution frames are read from HDF5, reduced to the evaluation
    repetitions (``_NREPS_EVAL``, dropping the first three) and stored with the
    same sharding + Blosc/bitshuffle codec pipeline.

    The array is built from one lazy block per Zarr chunk; each block is
    produced by :func:`_read_h5_batch`, which opens the HDF5 file on the worker
    and decodes only that batch, so memory stays bounded by concurrent tasks
    (the same argument as in :func:`bin_to_zarr`).

    Args:
        h5_path: Path to the HDF5 file written by :func:`bin_to_h5`.
        zarr_path: Path where the Zarr v3 store will be created
            (``...store.zarr/group/path/dataset_name``).
        dataset_name: Name of the source dataset in the HDF5 file. If ``None``,
            the file must contain exactly one dataset, which is used.

    Returns:
        Path to the zarr store.
    """
    h5_path = Path(h5_path)
    zarr_path = Path(zarr_path)

    store_path, group_path, output_dataset_name = _parse_zarr_path(zarr_path)

    # Read the layout metadata written by bin_to_h5 and resolve the source dataset.
    with h5py.File(h5_path, "r") as f:
        if dataset_name is None:
            dataset_name = _find_h5_dataset(f)
        ds = f[dataset_name]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"{dataset_name} is not a dataset in {h5_path}")
        raw_nreps = ds.shape[2]
        n_frames = ds.shape[0]
        if ds.shape[1] != _COLUMN_SIZE or ds.shape[3] != _ROW_SIZE:
            raise ValueError(
                f"Unexpected frame layout {ds.shape} in {h5_path}:{dataset_name}; "
                f"expected (n_frames, {_COLUMN_SIZE}, nreps, {_ROW_SIZE})"
            )

    eval_nreps = len(np.empty(raw_nreps)[_NREPS_EVAL])
    chunk_size = _frame_chunk_size(eval_nreps)

    array_shape = (n_frames, _COLUMN_SIZE, eval_nreps, _ROW_SIZE)
    chunk_shape = (chunk_size, _COLUMN_SIZE, eval_nreps, _ROW_SIZE)
    inner_chunk_shape = (chunk_size, 1, eval_nreps, _ROW_SIZE)

    # Construct full zarr array path
    full_array_path = (
        f"{store_path}/{group_path}/{output_dataset_name}" if group_path else f"{store_path}/{output_dataset_name}"
    )

    # We create the empty array structure here so we control the v3 codec
    # pipeline. Dask's store writes into it.
    target = zarr.open_array(
        full_array_path,
        mode="w",
        shape=array_shape,
        chunks=chunk_shape,
        dtype="uint16",
        zarr_format=3,
        codecs=_sharded_frame_codecs(inner_chunk_shape),
    )

    # Build one lazy Dask block per Zarr chunk along the frame axis. Each block
    # is produced on a worker by _read_h5_batch, which opens the HDF5 file and
    # decodes only that batch, so peak RAM scales with concurrent tasks.
    n_batches = (n_frames + chunk_size - 1) // chunk_size
    _logger.info(
        "Building Dask array of %d frames in %d chunks (chunk_size=%d frames)...",
        n_frames,
        n_batches,
        chunk_size,
    )

    blocks = []
    for batch_start in range(0, n_frames, chunk_size):
        batch_end = min(batch_start + chunk_size, n_frames)
        block = da.from_delayed(
            delayed(_read_h5_batch)(h5_path, dataset_name, batch_start, batch_end),
            shape=(batch_end - batch_start, _COLUMN_SIZE, eval_nreps, _ROW_SIZE),
            dtype=np.uint16,
        )
        blocks.append(block)

    data = da.concatenate(blocks, axis=0)

    # Stream the blocks into the pre-created compressed Zarr array. Writing into
    # the existing array object preserves our Blosc/bitshuffle codec pipeline.
    da.store(data, target, lock=False)

    _logger.info("Successfully wrote %d frames to %s", n_frames, zarr_path)
    return zarr_path


def get_chunk_info(zarr_path: str) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    z = zarr.open(zarr_path, mode="r")  # type: ignore[return-value]

    outer_chunk: tuple[int, ...] = z.metadata.chunk_grid.chunk_shape  # type: ignore[attr-defined]

    inner_chunk: tuple[int, ...] | None = None
    for codec in z.metadata.codecs:  # type: ignore[attr-defined]
        if hasattr(codec, "chunk_shape"):  # type: ignore[attr-defined]
            inner_chunk = codec.chunk_shape  # type: ignore
            break

    return outer_chunk, inner_chunk  # type: ignore


def print_store_info(zarr_path: str) -> None:
    """Print a combined tree and chunk layout summary for a zarr store."""

    def _lines(
        node: zarr.Array | zarr.Group,
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

    root = zarr.open(zarr_path, mode="r")
    lines = [zarr_path]
    if isinstance(root, zarr.Group):
        members = list(root.members())
        for i, (_, child) in enumerate(members):
            lines.extend(_lines(child, "", last=(i == len(members) - 1)))
    else:
        lines.extend(_lines(root, "", last=True))
    print("\n".join(lines))


def _rechunk_col_batch(
    source_store: str,
    source_array_path: str,
    target_store: str,
    target_array_path: str,
    col: int,
    col_batch: int,
    n_col: int,
    n_row: int,
) -> None:
    """Worker: read a ``col_batch``-wide column band, write single-pixel chunks.

    The band ``(n_frames, col_batch, n_reps, n_row)`` is the entire RAM
    footprint of the task. Each ``(n_frames, 1, n_reps, 1)`` target chunk is
    written individually so Zarr compresses one pixel at a time, avoiding any
    duplication of the band in parallel compression buffers.
    """
    source = zarr.open_array(source_store, path=source_array_path, mode="r")
    target = zarr.open_array(target_store, path=target_array_path, mode="r+")

    col_end = min(col + col_batch, n_col)
    band = np.asarray(source[:, col:col_end, :, :])  # (n_frames, col_batch, n_reps, n_row)

    for c in range(col_end - col):
        for j in range(n_row):
            # Each write targets exactly one pixel chunk -> conflict-free.
            target[:, col + c, :, j] = band[:, c, :, j]


def rechunk_to_pixels(
    source_path: str | Path,
    target_path: str | Path,
    col_batch: int = 1,
) -> None:
    """
    Rechunks a zarr store so that each chunk holds all frames and all readouts
    for a single spatial pixel. The target chunk shape is
    ``(n_frames, 1, n_reps, 1)`` — i.e. one chunk per (column, row) pixel.

    The source is expected to have shape ``(frames, 64, nreps, 64)``.

    Source shards cover the full spatial extent, so every shard must be
    decompressed regardless of how many columns are requested. Each task reads
    ``col_batch`` columns and writes their pixel chunks. Smaller ``col_batch``
    means lower peak RAM per worker but more passes over (re-decompressions of)
    the source. ``col_batch=1`` minimises memory. Independent column bands are
    rechunked in parallel via Dask.

    Memory per worker (the dominant term) is the column band held in RAM::

        peak_band_bytes ≈ n_frames * col_batch * n_reps * n_row * itemsize

    For example, n_frames=2000, n_reps=50, n_row=64, uint16 (2 bytes),
    col_batch=1::

        2000 * 1 * 50 * 64 * 2  ≈ 1.22 GB

    Doubling col_batch doubles this. Total concurrent RAM across the cluster is
    roughly ``n_workers * peak_band_bytes`` plus modest Zarr write buffers.

    Args:
        source_path: Path to the source zarr store (chunked along frames).
        target_path: Path where the rechunked zarr store will be written.
        col_batch: Number of columns read per task. Controls the memory/IO
            trade-off; 1 gives the smallest footprint.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    # Parse source path to extract store and array path
    source_str = str(source_path)
    if ".zarr" in source_str:
        store_end = source_str.find(".zarr") + len(".zarr")
        source_store = source_str[:store_end]
        source_array_path = source_str[store_end:].lstrip("/")
        source = cast(zarr.Array, zarr.open_array(source_store, path=source_array_path, mode="r"))  # type: ignore
    else:
        source_store = source_str
        source_array_path = ""
        source = cast(zarr.Array, zarr.open(source_str, mode="r"))

    n_frames, n_col, n_reps, n_row = source.shape

    # Parse target path to extract store and array path
    target_str = str(target_path)
    if ".zarr" in target_str:
        store_end = target_str.find(".zarr") + len(".zarr")
        target_store = target_str[:store_end]
        target_array_path = target_str[store_end:].lstrip("/")
        # create empty array with single-pixel chunks
        zarr.open_array(
            target_store,
            path=target_array_path,
            mode="w",
            shape=source.shape,
            chunks=(n_frames, 1, n_reps, 1),
            dtype=source.dtype,
            zarr_format=3,
            codecs=_blosc_codecs(),
        )  # type: ignore
    else:
        target_store = target_str
        target_array_path = ""
        # create empty array with single-pixel chunks
        zarr.open_array(
            target_store,
            mode="w",
            shape=source.shape,
            chunks=(n_frames, 1, n_reps, 1),
            dtype=source.dtype,
            zarr_format=3,
            codecs=_blosc_codecs(),
        )

    tasks = []
    for i in range(0, n_col, col_batch):
        task = delayed(_rechunk_col_batch)(
            source_store, source_array_path, target_store, target_array_path, i, col_batch, n_col, n_row
        )
        tasks.append(task)

    _logger.info("Executing %d Dask tasks to rechunk %s...", len(tasks), source_path)
    compute(*tasks)

    _logger.info("Rechunked %s -> %s", source_path, target_path)


def compute_median(data_p: da.Array, path: str | Path) -> None:
    median_array = da.median(data_p, axis=(0, 2))
    # rechunk to a single chunk and write to zarr
    median_array.rechunk(-1).to_zarr(path)


def compute_offset_corr(data_f: da.Array, median: da.Array, path: str | Path) -> None:
    # Rechunk axis 1 (columns) to a single block so downstream output drops the
    # inner column chunks and processes full frames.
    data_f = data_f.rechunk(cast(Any, {1: -1}))
    offset_corr_array = data_f - median[np.newaxis, :, np.newaxis, :]
    offset_corr_array.to_zarr(path)


def compute_common_modes(data: da.Array, path: str | Path) -> None:
    common_modes_array = da.median(data, axis=3)
    common_modes_array.to_zarr(path)


def compute_slopes(data: da.Array, path: str | Path) -> None:
    # shape is (frames, columns, nreps, rows)
    n = data.shape[2]  # nreps is axis 2
    x = np.arange(n, dtype=np.float64)  # plain NumPy, tiny
    x_dev = x - x.mean()  # plain NumPy
    denominator = float((x_dev**2).sum())  # scalar, computed now

    # Multiply along axis 2 (nreps), then sum along axis 2
    # x_dev shape must broadcast to (1, 1, nreps, 1)
    slopes_array = (data * x_dev[np.newaxis, np.newaxis, :, np.newaxis]).sum(axis=2) / denominator
    slopes_array.to_zarr(path)


def subtract(data: da.Array, common_modes: da.Array, path: str | Path) -> None:
    signals_array = data - common_modes[:, :, :, np.newaxis]
    signals_array.to_zarr(path)


def compute_msd(data: da.Array, median: da.Array, path: str | Path) -> None:
    """
    Computes the Mean Squared Deviation for each pixel.

    Args:
        data: Dask array of shape (frames, 64, nreps, 64)
        median: Dask/NumPy array of shape (64, 64)
        path: Path to save the resulting (64, 64) array
    """
    # 1. Align median for broadcasting: (64, 64) -> (1, 64, 1, 64)
    # Axis 0 (frames) and Axis 2 (nreps) are new dimensions
    median_aligned = median[np.newaxis, :, np.newaxis, :]

    # 2. Calculate squared differences
    # Resulting shape: (frames, 64, nreps, 64)
    squared_diff = (data - median_aligned) ** 2

    # 3. Average over frames (axis 0) and repetitions (axis 2)
    # Resulting shape: (64, 64)
    msd_array = da.mean(squared_diff, axis=(0, 2))

    # 4. Store the result
    # We rechunk to -1 because the output is tiny (64x64)
    msd_array.rechunk(-1).to_zarr(path)


def compute_signals_mean(signals: da.Array, path: str | Path) -> None:
    signals_median_array = da.mean(signals, axis=2)
    signals_median_array.to_zarr(path)
