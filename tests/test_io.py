"""Tests for apantias.io helpers and zarr utilities."""

# Tests import package-internal helpers and exercise Zarr's runtime API, which
# has incomplete stubs in strict mode.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from pathlib import Path

import numpy as np
import pytest
import zarr

from apantias.io._helpers import (
    _frame_chunk_size,
    _parse_bin_frames,
    _read_frame_batch,
    _resolve_range_indices,
)
from apantias.io.zarr import _blosc_codecs, _parse_zarr_path, get_chunk_info
from apantias.settings import RangeSpec


def test_resolve_range_indices() -> None:
    spec = RangeSpec(start=0, stop=5, step=1)
    assert _resolve_range_indices(spec, 10) == [0, 1, 2, 3, 4]


def test_frame_chunk_size() -> None:
    # bytes_per_frame = 64 * 200 * 64 * 2 = 1,638,400
    # target 100MB => ~61 frames, rounded down to nearest 10 -> 60
    assert _frame_chunk_size(200, 64, 64, 100 * 1024 * 1024) == 60
    # Tiny target falls back to 1
    assert _frame_chunk_size(200, 64, 64, 1) == 1


def test_parse_zarr_path() -> None:
    store, group, ds = _parse_zarr_path("/tmp/data.zarr/group/dataset")
    assert store == "/tmp/data.zarr"
    assert group == "group"
    assert ds == "dataset"


def test_parse_zarr_path_no_group() -> None:
    store, group, ds = _parse_zarr_path("/tmp/data.zarr/dataset")
    assert store == "/tmp/data.zarr"
    assert group is None
    assert ds == "dataset"


def test_parse_zarr_path_missing_dot_zarr() -> None:
    with pytest.raises(ValueError, match=".zarr"):
        _parse_zarr_path("/tmp/data.h5/dataset")


def test_get_chunk_info(tmp_path: Path) -> None:
    path = tmp_path / "test.zarr"
    zarr.open_array(
        str(path),
        mode="w",
        shape=(10, 8, 4, 8),
        chunks=(5, 8, 4, 8),
        dtype="uint16",
        zarr_format=3,
        codecs=_blosc_codecs(),
    )
    outer, inner = get_chunk_info(str(path))
    assert outer == (5, 8, 4, 8)
    assert inner is None


def test_parse_bin_frames(tmp_path: Path) -> None:
    """Build a minimal binary file with 2 frames and verify parsing.

    Frame format: each frame spans ``lines_per_frame`` data lines. The key
    sentinel (65535) lives in the key word right after the ``cols`` pixel
    words and marks the *first* line of each frame. The next frame key appears
    ``lines_per_frame`` lines later, and that key line is also the last data
    line of the current frame.
    """
    # rows=2, nreps=3, cols=2 -> lines_per_frame = 6, raw_row_size = 3
    rows, nreps, cols, key_ints = 2, 3, 2, 1
    lines_per_frame = rows * nreps  # 6
    bin_file = tmp_path / "test.bin"

    data_rows: list[list[int]] = []
    for frame in range(2):
        for i in range(lines_per_frame):
            # The first line of each frame block is the key line.
            if i == 0:
                data_rows.append([0, 0, 65535])
            else:
                line = i - 1
                rep = line % nreps
                row = line // nreps
                data_rows.append([rep, row, frame * 10 + row])
    # Add the final key line that terminates the last frame.
    data_rows.append([0, 0, 65535])

    arr = np.array(data_rows, dtype="uint16")
    arr.tofile(bin_file)

    starts, ends, n_frames = _parse_bin_frames(bin_file, offset=0, nreps=nreps, rows=rows, cols=cols, key_ints=key_ints)
    assert n_frames == 2
    assert len(starts) == 2
    assert len(ends) == 2
    assert list(ends - starts) == [lines_per_frame, lines_per_frame]


def test_parse_bin_frames_asymmetric(tmp_path: Path) -> None:
    """Regression: the key sentinel must be located after the ``cols`` pixel
    words. This only differs from other word positions when rows != cols, so
    the test uses asymmetric dimensions.
    """
    rows, nreps, cols, key_ints = 3, 2, 4, 1
    lines_per_frame = rows * nreps  # 6
    bin_file = tmp_path / "asymmetric.bin"

    data_rows: list[list[int]] = []
    for frame in range(2):
        for i in range(lines_per_frame):
            if i == 0:
                data_rows.append([0] * cols + [65535])
            else:
                data_rows.append([0] * cols + [frame * 100 + i])
    data_rows.append([0] * cols + [65535])

    arr = np.array(data_rows, dtype="uint16")
    arr.tofile(bin_file)

    starts, ends, n_frames = _parse_bin_frames(bin_file, offset=0, nreps=nreps, rows=rows, cols=cols, key_ints=key_ints)
    assert n_frames == 2
    assert list(ends - starts) == [lines_per_frame, lines_per_frame]


def test_read_frame_batch_asymmetric(tmp_path: Path) -> None:
    """Verify the raw-to-4D mapping for asymmetric dimensions.

    Layout contract (see README): output shape is
    (n_frames, n_rows, n_reps, n_cols); a raw line holds the ``cols`` pixel
    values of one readout, and consecutive lines cycle through the
    repetitions (axis 2) fastest, then rows (axis 1).
    """
    rows, nreps, cols, key_ints = 3, 2, 4, 1
    lines_per_frame = rows * nreps  # 6
    raw_row_size = cols + key_ints  # 5
    bin_file = tmp_path / "batch.bin"

    data_rows: list[list[int]] = []
    for i in range(lines_per_frame):
        if i == 0:
            data_rows.append([0] * cols + [65535])
        else:
            line = i - 1
            rep = line % nreps
            row = line // nreps
            pixels = [rep * 100 + row * 10 + w for w in range(cols)]
            data_rows.append(pixels + [0])
    # The terminating key line doubles as the last data line of the frame:
    # its pixel words carry real values, the sentinel sits in the key word.
    last_rep = (lines_per_frame - 1) % nreps
    last_row = (lines_per_frame - 1) // nreps
    data_rows.append([last_rep * 100 + last_row * 10 + w for w in range(cols)] + [65535])

    arr = np.array(data_rows, dtype="uint16")
    arr.tofile(bin_file)

    starts = np.array([0], dtype=np.intp)
    ends = np.array([lines_per_frame], dtype=np.intp)
    out = _read_frame_batch(
        bin_file, 0, starts, ends, 0, 1, nreps, slice(None), cols=cols, rows=rows, raw_row_size=raw_row_size
    )

    assert out.shape == (1, rows, nreps, cols)
    for rep in range(nreps):
        for row in range(rows):
            for w in range(cols):
                assert out[0, row, rep, w] == rep * 100 + row * 10 + w
