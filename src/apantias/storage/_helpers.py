# pyright: reportUnusedFunction=false
# The functions in this module are intentionally small, reusable helpers that are
# imported by the other I/O submodules and exercised directly by unit tests. They
# are not called inside this file, so Pyright's strict unused-function check is
# disabled here rather than adding noise to every helper.

"""Private helpers shared across I/O modules."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from apantias.settings import RangeSpec


def _resolve_range_indices(
    spec: RangeSpec,
    size: int,
) -> list[int]:
    """Resolve a RangeSpec into a list of concrete indices.

    Negative indices in *spec* are resolved relative to *size* using
    Python's standard range semantics (e.g. stop=-1  →  size-1).
    Out-of-bounds indices are silently kept — the caller should
    filter them if the underlying data is smaller than expected.
    """
    return list(range(spec.start, spec.resolve_stop(size), spec.step))


def _frame_chunk_size(nreps: int, rows: int, cols: int, target_chunk_bytes: int) -> int:
    """Number of frames per write/compression chunk for the given ``nreps``.

    Bounded so each chunk is at most ``target_chunk_bytes``, then
    rounded down to the nearest multiple of 10 (or kept as 1 if tiny), so the
    same write-unit budget is used for the HDF5 and Zarr pipelines.
    """
    bytes_per_frame = rows * nreps * cols * 2
    chunk_size = max(1, target_chunk_bytes // bytes_per_frame)
    if chunk_size >= 10:
        chunk_size = (chunk_size // 10) * 10
    return chunk_size


def _read_frame_batch(
    bin_file: Path,
    offset: int,
    frame_start_indices: npt.NDArray[np.intp],
    frame_end_indices: npt.NDArray[np.intp],
    batch_start: int,
    batch_end: int,
    nreps: int,
    nreps_slice: slice,
    rows: int,
    cols: int,
    raw_row_size: int,
) -> npt.NDArray[np.uint16]:
    """Extract one batch of frames from the binary file as a NumPy array."""
    raw_uint16 = np.memmap(bin_file, dtype="uint16", mode="r", offset=offset)
    n_complete_rows = len(raw_uint16) // raw_row_size
    raw_data = raw_uint16[: n_complete_rows * raw_row_size].reshape(-1, raw_row_size)

    batch = np.stack([
        raw_data[frame_start_indices[i] + 1 : frame_end_indices[i] + 1, :cols] for i in range(batch_start, batch_end)
    ])
    # Reshape to (n_frames, n_rows, n_reps, n_cols): a raw line holds the
    # ``cols`` pixel values of one readout, and consecutive lines cycle
    # through the repetitions (axis 2) fastest, then rows (axis 1).
    batch_full = batch.reshape(-1, rows, nreps, cols)
    return batch_full[:, :, nreps_slice, :]


def _parse_bin_frames(
    bin_file: Path, offset: int, nreps: int, rows: int, cols: int, key_ints: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
    """Locate the valid frame boundaries in a binary file."""
    lines_per_frame = rows * nreps
    raw_row_size = cols + key_ints
    raw_uint16 = np.memmap(bin_file, dtype="uint16", mode="r", offset=offset)
    n_complete_rows = len(raw_uint16) // raw_row_size
    raw_data = raw_uint16[: n_complete_rows * raw_row_size].reshape(-1, raw_row_size)

    # Locate all frame-key rows: the sentinel 65535 sits in the key word right
    # after the ``cols`` pixel words of a line.
    frame_key_positions = np.where(raw_data[:, cols] == 65535)[0]
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
