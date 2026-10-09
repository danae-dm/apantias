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

from apantias.settings import AnalysisSettings, FrameSettings, RangeSpec

_logger = logging.getLogger(__name__)


def _resolve_range_indices(
    spec: "RangeSpec",
    size: int,
) -> list[int]:
    """Resolve a RangeSpec into a list of concrete indices.

    Negative indices in *spec* are resolved relative to *size* using
    Python's standard range semantics (e.g. stop=-1  →  size-1).
    Out-of-bounds indices are silently kept — the caller should
    filter them if the underlying data is smaller than expected.
    """
    return list(range(spec.start, spec.stop, spec.step))


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

    The Zarr chunk (= the write/transfer unit, ``target_chunk_mb``) becomes a *shard* that
    is internally subdivided into ``inner_chunk_shape`` inner chunks, each
    compressed separately with :func:`_blosc_codecs`.

    Making the inner chunk one row high ``(chunk_size, 1, n_reps, n_cols)`` is
    what accelerates the downstream pixel rechunk: reading a single row then
    becomes a *partial shard read* that decompresses only that row's inner
    chunks instead of the whole shard. Across all 64 row passes every inner
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
    n_reps: int,
    rep_slice: slice,
    n_rows: int,
    n_cols: int,
    raw_line_size: int,
) -> np.ndarray:
    """Extract one batch of frames from the binary file as a NumPy array.

    This is the per-chunk worker for the Dask array. It memory-maps the file
    (only the pages for *this* batch are paged in by the OS), gathers the frame
    lines for ``batch_start:batch_end`` and returns a
    ``(batch, n_rows, eval_n_reps, n_cols)`` array. The only RAM it holds is
    that single decoded batch, so total memory is bounded by the number of
    Dask tasks running concurrently, not by the file size.
    """

    raw_uint16 = np.memmap(bin_file, dtype="uint16", mode="r", offset=offset)
    n_complete_lines = len(raw_uint16) // raw_line_size
    raw_data = raw_uint16[: n_complete_lines * raw_line_size].reshape(-1, raw_line_size)

    batch = np.stack([
        raw_data[frame_start_indices[frame_idx] + 1 : frame_end_indices[frame_idx] + 1, :n_cols]
        for frame_idx in range(batch_start, batch_end)
    ])
    # Reshape to (frame_idx, row_idx, rep_idx, col_idx), apply the rep slice, then return
    batch_full = batch.reshape(-1, n_rows, n_reps, n_cols)
    return batch_full[:, :, rep_slice, :]


def _frame_chunk_size(n_reps: int, n_rows: int, n_cols: int, target_chunk_mb: int, n_frames: int) -> int:
    """Number of frames per write/compression chunk for the given ``n_reps``.

    Bounded so each chunk is at most ``target_chunk_mb`` MB, then
    rounded down to the nearest multiple of 10 (or kept as 1 if tiny), so the
    same write-unit budget is used for the HDF5 and Zarr pipelines. Finally
    capped at ``n_frames``, since HDF5 rejects chunks larger than the dataset.
    """
    bytes_per_frame = n_rows * n_reps * n_cols * 2
    chunk_size = max(1, (target_chunk_mb * 1024 * 1024) // bytes_per_frame)
    if chunk_size >= 10:
        chunk_size = (chunk_size // 10) * 10
    return max(1, min(chunk_size, n_frames))


def _parse_bin_frames(
    bin_file: Path, offset: int, n_reps: int, n_rows: int, n_cols: int, key_ints: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Locate the valid frame boundaries in a binary file.

    Each line of the binary file holds the ``n_cols`` pixel values of one sensor
    row readout followed by ``key_ints`` key integers.

    Memory-maps the raw file (only the pages actually touched are loaded by the
    OS), finds every frame-key line (sentinel 65535 at position ``n_cols``)
    and returns the start/end line indices of consecutive key pairs whose
    interiors span exactly ``n_rows * n_reps`` lines, plus the frame count.
    """
    lines_per_frame = n_rows * n_reps
    raw_line_size = n_cols + key_ints
    raw_uint16 = np.memmap(bin_file, dtype="uint16", mode="r", offset=offset)
    n_complete_lines = len(raw_uint16) // raw_line_size
    raw_data = raw_uint16[: n_complete_lines * raw_line_size].reshape(-1, raw_line_size)

    # Locate all frame-key lines (sentinel 65535 right after the pixel values).
    frame_key_positions = np.where(raw_data[:, n_cols] == 65535)[0]
    if len(frame_key_positions) < 2:
        raise ValueError(f"No valid frames found in {bin_file}")
    # A valid frame has exactly lines_per_frame lines between two consecutive keys.
    starts = frame_key_positions[:-1]
    ends = frame_key_positions[1:]
    valid_mask = (ends - starts) == lines_per_frame
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


def bin_to_h5(
    bin_path: str | Path, n_reps: int, h5_path: str | Path, dataset_name: str, frame: FrameSettings, offset: int = 8
) -> Path:
    """Write the frames of a binary file to a compressed HDF5 dataset.

    The no-Dask counterpart of :func:`bin_to_zarr`: it parses the same frame
    boundaries (via :func:`_parse_bin_frames`) but keeps **all** repetitions and
    writes the full-resolution data as a uint16 dataset of shape
    ``(n_frames, n_rows, n_reps, n_cols)``. Dropping the first three
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
    ``frame.target_chunk_mb`` regardless of the number of frames. The metadata
    needed by :func:`h5_to_zarr` (``n_reps``, frame count, layout) is stored as
    attributes on the dataset.

    Args:
        bin_path: Path to the source .bin file containing uint16 values.
        n_reps: Number of repetitions per frame row.
        h5_path: Path where the HDF5 file will be created (its parent directory
            must exist).
        dataset_name: Name of the dataset within the HDF5 file.
        offset: Byte offset into the binary file to start reading from.

    Returns:
        Path to the created HDF5 file.
    """
    bin_path = Path(bin_path)
    h5_path = Path(h5_path)

    frame_start_indices, frame_end_indices, n_frames = _parse_bin_frames(
        bin_path, offset, n_reps, frame.n_rows, frame.n_cols, frame.key_ints
    )
    _logger.info("Found %d valid frames in %s", n_frames, bin_path)

    chunk_size = _frame_chunk_size(n_reps, frame.n_rows, frame.n_cols, frame.target_chunk_mb, n_frames)
    filters = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=1)
    shape = (n_frames, frame.n_rows, n_reps, frame.n_cols)
    chunk_shape = (chunk_size, frame.n_rows, n_reps, frame.n_cols)

    # Keep all reps (no slice)
    rep_slice = slice(None)

    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset(dataset_name, shape=shape, dtype="uint16", chunks=chunk_shape, **filters)
        ds.attrs["nreps"] = n_reps
        ds.attrs["offset"] = offset
        ds.attrs["n_frames"] = n_frames
        ds.attrs["column_size"] = frame.n_cols
        ds.attrs["row_size"] = frame.n_rows
        ds.attrs["raw_row_size"] = frame.n_cols + frame.key_ints

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
                    n_reps,
                    rep_slice,
                    frame.n_rows,
                    frame.n_cols,
                    frame.n_cols + frame.key_ints,
                )
            )

        # Execute in parallel (threads are fine here — GIL released by
        # numpy/hdf5plugin under the hood)
        compute(*delayed_tasks, scheduler="threads")

    return h5_path


def _write_h5_batch(
    ds,
    start,
    end,
    bin_path,
    offset,
    frame_start_indices,
    frame_end_indices,
    n_reps,
    rep_slice,
    n_rows,
    n_cols,
    raw_line_size,
):
    """Read one batch from binary and write it to a pre-opened h5py dataset."""
    batch = _read_frame_batch(
        bin_path,
        offset,
        frame_start_indices,
        frame_end_indices,
        start,
        end,
        n_reps,
        rep_slice,
        n_rows,
        n_cols,
        raw_line_size,
    )
    ds[start:end] = batch


def _read_h5_batch(
    h5_path: Path,
    dataset_name: str,
    batch_start: int,
    batch_end: int,
    frame_indices: list[int] | None = None,
    rep_indices: list[int] | None = None,
) -> np.ndarray:
    """Read one batch of frames from an HDF5 dataset as a NumPy array.

    Optional *frame_indices* and *rep_indices* select specific positions
    within the batch / rep dimension.  When *frame_indices* is given it
    should contain indices into the original (un-batched) frame axis;
    they are translated to batch-relative indices before use.

    Returns ``(n_frames, n_rows, n_reps, n_cols)``.
    """
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"{dataset_name} is not a dataset in {h5_path}")
        batch = ds[batch_start:batch_end]

    # Translate frame indices from original-axis → batch-relative.
    # Keep only those that fall within the actual batch.
    if frame_indices is not None:
        batch_frame_indices = [
            frame_idx - batch_start for frame_idx in frame_indices if 0 <= frame_idx - batch_start < len(batch)
        ]
        if batch_frame_indices:
            batch = batch[batch_frame_indices]

    # Rep indices are applied directly (no translation needed).
    if rep_indices is not None:
        batch = batch[:, :, rep_indices, :]

    return batch


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
    frame: FrameSettings,
    analysis: AnalysisSettings,
    dataset_name: str | None = None,
) -> Path:
    """Write the data of an HDF5 dataset to a Zarr v3 store.

    The Dask-based counterpart of :func:`bin_to_zarr` for data that already
    went through :func:`bin_to_h5`.  It produces the same store that
    ``bin_to_zarr`` would: full-resolution frames are read from HDF5,
    reduced to the selected repetitions and stored with the same sharding +
    Blosc/bitshuffle codec pipeline.

    Args:
        h5_path:       Path to the HDF5 file written by :func:`bin_to_h5`.
        zarr_path:     Path where the Zarr v3 store will be created.
        dataset_name:  Name of the source dataset in the HDF5 file.
        nreps_range:   RangeSpec for selecting repetitions.
                       If ``None``, all reps are used.
        frames_range:  RangeSpec for selecting frame indices.
                       If ``None``, all frames are included.

    Returns:
        Path to the zarr store.
    """

    h5_path = Path(h5_path)
    zarr_path = Path(zarr_path)

    store_path, group_path, output_dataset_name = _parse_zarr_path(zarr_path)

    # Read the layout metadata and resolve the source dataset.
    with h5py.File(h5_path, "r") as f:
        if dataset_name is None:
            dataset_name = _find_h5_dataset(f)
        ds = f[dataset_name]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"{dataset_name} is not a dataset in {h5_path}")
        n_frames, n_rows, raw_n_reps, n_cols = ds.shape
        if n_rows != frame.n_rows or n_cols != frame.n_cols:
            raise ValueError(
                f"Unexpected frame layout {ds.shape} in {h5_path}:{dataset_name}; "
                f"expected (n_frames, {frame.n_rows}, n_reps, {frame.n_cols})"
            )

    # Resolve rep indices from analysis settings
    rep_indices = _resolve_range_indices(analysis.nreps_range, raw_n_reps)
    eval_n_reps = len(rep_indices)

    # Resolve frame indices from analysis settings
    frame_indices = _resolve_range_indices(analysis.frames_range, n_frames)
    n_output_frames = len(frame_indices)

    chunk_size = _frame_chunk_size(eval_n_reps, n_rows, n_cols, frame.target_chunk_mb, n_output_frames)

    array_shape = (n_output_frames, n_rows, eval_n_reps, n_cols)
    chunk_shape = (chunk_size, n_rows, eval_n_reps, n_cols)
    inner_chunk_shape = (chunk_size, 1, eval_n_reps, n_cols)

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
        "Building Dask array of %d frames (%d reps) in %d chunks (chunk_size=%d frames)...",
        n_output_frames,
        eval_n_reps,
        n_batches,
        chunk_size,
    )

    blocks = []
    for batch_start in range(0, n_output_frames, chunk_size):
        batch_end = min(batch_start + chunk_size, n_output_frames)

        # Translate output-frame offsets to actual HDF5 frame indices
        h5_start = frame_indices[batch_start]
        h5_end = frame_indices[batch_end] if batch_end < n_output_frames else frame_indices[-1] + 1

        block = da.from_delayed(
            delayed(_read_h5_batch)(
                h5_path,
                dataset_name,
                h5_start,  # ← was batch_start
                h5_end,  # ← was batch_end
                frame_indices=frame_indices,
                rep_indices=rep_indices,
            ),
            shape=(batch_end - batch_start, n_rows, eval_n_reps, n_cols),
            dtype=np.uint16,
        )
        blocks.append(block)

    data = da.concatenate(blocks, axis=0)
    da.store(data, target, lock=False)

    _logger.info("Successfully wrote %d frames to %s", n_output_frames, zarr_path)
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


def _rechunk_row_batch(
    source_store: str,
    source_array_path: str,
    target_store: str,
    target_array_path: str,
    row_start: int,
    row_batch: int,
    n_rows: int,
    n_cols: int,
) -> None:
    """Worker: read a ``row_batch``-high row band, write single-pixel chunks.

    The band ``(n_frames, row_batch, n_reps, n_cols)`` is the entire RAM
    footprint of the task. Each ``(n_frames, 1, n_reps, 1)`` target chunk is
    written individually so Zarr compresses one pixel at a time, avoiding any
    duplication of the band in parallel compression buffers.
    """
    source = zarr.open_array(source_store, path=source_array_path, mode="r")
    target = zarr.open_array(target_store, path=target_array_path, mode="r+")

    row_end = min(row_start + row_batch, n_rows)
    band = np.asarray(source[:, row_start:row_end, :, :])  # (n_frames, row_batch, n_reps, n_cols)

    for row_idx in range(row_start, row_end):
        for col_idx in range(n_cols):
            # Each write targets exactly one pixel chunk -> conflict-free.
            target[:, row_idx, :, col_idx] = band[:, row_idx - row_start, :, col_idx]


def rechunk_to_pixels(
    source_path: str | Path,
    target_path: str | Path,
    row_batch: int = 1,
) -> None:
    """
    Rechunks a zarr store so that each chunk holds all frames and all readouts
    for a single spatial pixel. The target chunk shape is
    ``(n_frames, 1, n_reps, 1)`` — i.e. one chunk per (row, column) pixel.

    The source is expected to have shape ``(n_frames, n_rows, n_reps, n_cols)``.

    Source shards cover the full spatial extent, so every shard must be
    decompressed regardless of how many rows are requested. Each task reads
    ``row_batch`` rows and writes their pixel chunks. Smaller ``row_batch``
    means lower peak RAM per worker but more passes over (re-decompressions of)
    the source. ``row_batch=1`` minimises memory. Independent row bands are
    rechunked in parallel via Dask.

    Memory per worker (the dominant term) is the row band held in RAM::

        peak_band_bytes ≈ n_frames * row_batch * n_reps * n_cols * itemsize

    For example, n_frames=2000, n_reps=50, n_cols=64, uint16 (2 bytes),
    row_batch=1::

        2000 * 1 * 50 * 64 * 2  ≈ 1.22 GB

    Doubling row_batch doubles this. Total concurrent RAM across the cluster is
    roughly ``n_workers * peak_band_bytes`` plus modest Zarr write buffers.

    Args:
        source_path: Path to the source zarr store (chunked along frames).
        target_path: Path where the rechunked zarr store will be written.
        row_batch: Number of rows read per task. Controls the memory/IO
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

    n_frames, n_rows, n_reps, n_cols = source.shape

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
    for row_start in range(0, n_rows, row_batch):
        task = delayed(_rechunk_row_batch)(
            source_store, source_array_path, target_store, target_array_path, row_start, row_batch, n_rows, n_cols
        )
        tasks.append(task)

    _logger.info("Executing %d Dask tasks to rechunk %s...", len(tasks), source_path)
    compute(*tasks)

    _logger.info("Rechunked %s -> %s", source_path, target_path)


def compute_median(data_p: da.Array, path: str | Path) -> None:
    # median over frame_idx (axis 0) and rep_idx (axis 2) -> (n_rows, n_cols)
    median_array = da.median(data_p, axis=(0, 2))
    # rechunk to a single chunk and write to zarr
    median_array.rechunk(-1).to_zarr(path)


def compute_offset_corr(data_f: da.Array, median: da.Array, path: str | Path) -> None:
    # Rechunk axis 1 (row_idx) to a single block so downstream output drops the
    # inner row chunks and processes full frames.
    data_f = data_f.rechunk(cast(Any, {1: -1}))
    offset_corr_array = data_f - median[np.newaxis, :, np.newaxis, :]
    offset_corr_array.to_zarr(path)


def compute_common_modes(data: da.Array, path: str | Path) -> None:
    # median over col_idx (axis 3) -> (n_frames, n_rows, n_reps)
    common_modes_array = da.median(data, axis=3)
    common_modes_array.to_zarr(path)


def compute_slopes(data: da.Array, path: str | Path) -> None:
    # shape is (n_frames, n_rows, n_reps, n_cols)
    n_reps = data.shape[2]  # rep_idx is axis 2
    x = np.arange(n_reps, dtype=np.float64)  # plain NumPy, tiny
    x_dev = x - x.mean()  # plain NumPy
    denominator = float((x_dev**2).sum())  # scalar, computed now

    # Multiply along axis 2 (rep_idx), then sum along axis 2
    # x_dev shape must broadcast to (1, 1, n_reps, 1)
    slopes_array = (data * x_dev[np.newaxis, np.newaxis, :, np.newaxis]).sum(axis=2) / denominator
    slopes_array.to_zarr(path)


def subtract(data: da.Array, common_modes: da.Array, path: str | Path) -> None:
    # common_modes (n_frames, n_rows, n_reps) is broadcast along col_idx (axis 3)
    signals_array = data - common_modes[:, :, :, np.newaxis]
    signals_array.to_zarr(path)


def compute_signals_mean(signals: da.Array, path: str | Path) -> None:
    # mean over rep_idx (axis 2) -> (n_frames, n_rows, n_cols)
    signals_median_array = da.mean(signals, axis=2)
    signals_median_array.to_zarr(path)


def apply_pixelwise(data: da.Array, path: str | Path, func) -> None:
    """Apply a custom function to each pixel's frames and repetitions.

    Parameters
    ----------
    data : da.Array
        Pixelwise data of shape (n_frames, n_rows, n_reps, n_cols), chunked
        as (n_frames, 1, n_reps, 1) — one chunk per pixel.
    path : str or Path
        Output zarr path. The result has shape (n_rows, n_cols).
    func : callable
        User function that takes a 2-D array of shape (n_frames, n_reps) and
        returns a single scalar value. Applied independently to each
        pixel without loading the full array into memory.
    """

    def _apply_and_squeeze(chunk):
        return np.asarray(func(chunk.squeeze(axis=(1, 3)))).reshape(1, 1)

    result = da.map_blocks(_apply_and_squeeze, data, dtype=float, drop_axis=[0, 2])
    result.to_zarr(path)
