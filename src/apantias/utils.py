import logging
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import numpy as np
import zarr
from zarr.codecs import (
    BloscCodec,
    BloscShuffle,
    BytesCodec,
)
from typing import cast
import dask
import dask.array as da

_logger = logging.getLogger(__name__)

_COLUMN_SIZE = 64
_ROW_SIZE = 64
_KEY_INTS = 3
_RAW_ROW_SIZE = _ROW_SIZE + _KEY_INTS  # 67 uint16 values per raw row
_TARGET_CHUNK_BYTES = 200 * 1024 * 1024  # 100 MB
_NREPS_EVAL = [-1]  # TODO: implement this


def _extract_and_write_batch(
    bin_file: Path,
    zarr_path: Path,
    offset: int,
    batch_start: int,
    batch_end: int,
    frame_start_indices: np.ndarray,
    frame_end_indices: np.ndarray,
    nreps: int,
) -> None:
    """Worker function to extract a batch of frames from the binary file and write directly to Zarr."""
    # We use a memory map per worker to read the binary file concurrently
    raw_uint16 = np.memmap(bin_file, dtype="uint16", mode="r", offset=offset)
    n_complete_rows = len(raw_uint16) // _RAW_ROW_SIZE
    raw_data = raw_uint16[: n_complete_rows * _RAW_ROW_SIZE].reshape(-1, _RAW_ROW_SIZE)

    batch = np.array([
        raw_data[frame_start_indices[i] + 1 : frame_end_indices[i] + 1, :_ROW_SIZE]
        for i in range(batch_start, batch_end)
    ])
    # Reshape (batch, rows_per_frame, ROW_SIZE) → (batch, COLUMN_SIZE, nreps, ROW_SIZE)
    batch = batch.reshape(-1, _COLUMN_SIZE, nreps, _ROW_SIZE)

    # Open zarr array to write
    path_str = str(zarr_path)
    store_end = path_str.find(".zarr") + len(".zarr")
    store_path = path_str[:store_end]
    array_path = path_str[store_end:].lstrip("/")

    # Open the existing zarr array and write to the corresponding slice
    # Multiple processes can write to different chunks of a Zarr array simultaneously
    arr = zarr.open_array(store_path, path=array_path, mode="r+")
    arr[batch_start:batch_end] = batch


def bin_to_zarr(
    bin_file: str | Path,
    zarr_path: str | Path,
    nreps: int,
    offset: int = 8,
) -> Path:
    """
    Reads frames from a binary file and writes them to a Zarr v3 store.

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

    rows_per_frame = _COLUMN_SIZE * nreps
    bytes_per_frame = _COLUMN_SIZE * nreps * _ROW_SIZE * 2

    chunk_size = max(1, _TARGET_CHUNK_BYTES // bytes_per_frame)

    # Memory-map the raw file as uint16 — pages are loaded from disk on demand,
    # so the full file does not reside in RAM.
    raw_uint16 = np.memmap(bin_file, dtype="uint16", mode="r", offset=offset)
    n_complete_rows = len(raw_uint16) // _RAW_ROW_SIZE
    raw_data = raw_uint16[: n_complete_rows * _RAW_ROW_SIZE].reshape(-1, _RAW_ROW_SIZE)

    # Locate all frame-key rows (sentinel 65535 at column _COLUMN_SIZE).
    frame_key_positions = np.where(raw_data[:, _COLUMN_SIZE] == 65535)[0]
    if len(frame_key_positions) < 2:
        raise ValueError(f"No valid frames found in {bin_file}")
    # A valid frame has exactly rows_per_frame rows between two consecutive keys.
    # Create pairs of consecutive frame markers
    starts = frame_key_positions[:-1]
    ends = frame_key_positions[1:]
    # For each pair, compute spacing between markers
    valid_mask = (ends - starts) == rows_per_frame
    # Differences: [109-42, 176-109, 243-176] = [67, 67, 67]
    # Compare to rows_per_frame (64 * nreps)
    # → Boolean array: [True, True, True] (or False where spacing is wrong)
    # Keep only the valid start/end indices
    frame_start_indices = starts[valid_mask]
    frame_end_indices = ends[valid_mask]
    n_frames = len(frame_start_indices)

    if n_frames == 0:
        raise ValueError(f"No valid frames found in {bin_file}")

    _logger.info("Found %d valid frames in %s", n_frames, bin_file)
    # ------------------------------------------------------------------
    # Parse zarr_path to extract store, group, and dataset name.
    # Expected format: /path/to/store.zarr/group/path/dataset_name
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Zarr v3 array
    #   chunk (= compression unit): (chunk_size, 64, nreps, 64)
    # Codec pipeline per chunk:
    #   BytesCodec (little-endian uint16 bytes)
    #   → BloscCodec(zstd L9, bitshuffle)
    # ------------------------------------------------------------------
    array_shape = (n_frames, _COLUMN_SIZE, nreps, _ROW_SIZE)
    chunk_shape = (chunk_size, _COLUMN_SIZE, nreps, _ROW_SIZE)

    # Construct full zarr array path
    full_array_path = f"{store_path}/{group_path}/{dataset_name}" if group_path else f"{store_path}/{dataset_name}"

    # We just create the empty array structure here before launching workers
    zarr.open_array(
        full_array_path,
        mode="w",
        shape=array_shape,
        chunks=chunk_shape,
        dtype="uint16",
        zarr_format=3,
        codecs=[
            BytesCodec(endian="little"),
            BloscCodec(
                cname="zstd",
                clevel=9,
                shuffle=BloscShuffle.bitshuffle,
            ),
        ],
    )

    # Write frames in batches using Dask to distribute the work across cluster cores
    # This reads the binary file concurrently and writes directly to Zarr
    tasks = []
    for batch_start in range(0, n_frames, chunk_size):
        batch_end = min(batch_start + chunk_size, n_frames)

        # Schedule the batch extraction and writing as a Dask task
        task = dask.delayed(_extract_and_write_batch)(
            bin_file,
            zarr_path,
            offset,
            batch_start,
            batch_end,
            frame_start_indices,
            frame_end_indices,
            nreps,
        )
        tasks.append(task)

    _logger.info("Executing %d Dask tasks to convert binary to Zarr...", len(tasks))
    # Compute all tasks in parallel using the active Dask cluster
    dask.compute(*tasks)

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
            detail = [
                f"{child_prefix}shard : {outer}",
            ]
            if inner:
                detail.append(f"{child_prefix}chunk : {inner}")
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
    n_row: int,
) -> None:
    """Worker function to read a 2-column wide band and write 2x2 spatial blocks to Zarr."""
    source = zarr.open_array(source_store, path=source_array_path, mode="r")
    target = zarr.open_array(target_store, path=target_array_path, mode="r+")

    # Read just the 2 columns we need for this task (~440 MB footprint)
    band: np.ndarray = source[:, col : col + 2, :, :]  # (n_frames, 2, n_reps, n_row)

    # Write sequentially to avoid duplicating memory in ThreadPoolExecutor
    # and to prevent Zarr from spawning 32 parallel C compressions at once
    for j in range(0, n_row, 2):
        target[:, col : col + 2, :, j : j + 2] = band[:, :, :, j : j + 2]


def rechunk_to_pixels(
    source_path: str | Path,
    target_path: str | Path,
) -> None:
    """
    Rechunks a zarr store so that each chunk contains all frames and all
    readouts for a 2×2 block of spatial pixels (axis 1 and axis 3 each
    become size 2).

    The source is expected to have shape (frames, 64, nreps, 64).
    The resulting chunk shape is (target_frames_chunk, 2, nreps, 2), where
    target_frames_chunk is calculated to ensure chunks are ~100MB each.

    Source shards cover the full spatial extent, so every shard must be
    decompressed regardless of how many columns are requested. Reading
    ``col_batch`` columns per pass amortises that cost: the dataset is
    decompressed ``ceil(n_col / col_batch)`` times instead of
    ``ceil(n_col / 2)`` times. ``col_batch`` should be a multiple of 2.
    Target pixel chunks are written in parallel since they are independent.

    Peak RAM is roughly ``col_batch / n_col * dataset_size``.
    With the default col_batch=8 and a 64-column array that is ~1/8 of the
    uncompressed dataset size per pass.

    Args:
        source_path: Path to the source zarr store (chunked along frames).
        target_path: Path where the rechunked zarr store will be written.
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
        # create empty array
        zarr.open_array(
            target_store,
            path=target_array_path,
            mode="w",
            shape=source.shape,
            chunks=(n_frames, 2, n_reps, 2),
            dtype=source.dtype,
        )  # type: ignore
    else:
        target_store = target_str
        target_array_path = ""
        # create empty array
        zarr.open_array(
            target_store,
            mode="w",
            shape=source.shape,
            chunks=(n_frames, 2, n_reps, 2),
            dtype=source.dtype,
        )

    tasks = []
    for i in range(0, n_col, 2):
        task = dask.delayed(_rechunk_col_batch)(
            source_store, source_array_path, target_store, target_array_path, i, n_row
        )
        tasks.append(task)

    _logger.info("Executing %d Dask tasks to rechunk %s...", len(tasks), source_path)
    dask.compute(*tasks)

    _logger.info("Rechunked %s -> %s", source_path, target_path)


def compute_median(data_p: da.Array, path: str | Path) -> None:
    median_array = da.median(data_p, axis=(0, 2))
    median_array.rechunk(-1).astype(np.float32).to_zarr(path)


def compute_offset_corr(data_f: da.Array, median: da.Array, path: str | Path) -> None:
    offset_corr_array = data_f - median[np.newaxis, :, np.newaxis, :]
    offset_corr_array.astype(np.float32).to_zarr(path)


def compute_common_modes(data: da.Array, path: str | Path) -> None:
    common_modes_array = da.median(data, axis=3)
    common_modes_array.to_zarr(path)


def compute_slopes(data: da.Array, path: str | Path) -> None:
    n = data.shape[3]
    x = np.arange(n, dtype=np.float32)  # plain NumPy, tiny
    x_dev = x - x.mean()  # plain NumPy
    denominator = float((x_dev**2).sum())  # scalar, computed now

    # Single pass over offset_corr
    slopes_array = (data * x_dev[np.newaxis, np.newaxis, np.newaxis, :]).sum(axis=3) / denominator
    slopes_array.astype(np.float32).to_zarr(path)


def subtract(data: da.Array, common_modes: da.Array, path: str | Path) -> None:
    signals_array = data - common_modes[:, :, :, np.newaxis]
    signals_array.astype(np.float32).to_zarr(path)
