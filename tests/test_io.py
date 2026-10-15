"""Tests for apantias.io helpers and zarr utilities."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from apantias.io._helpers import (
    _frame_chunk_size,
    _parse_bin_frames,
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

    Frame format: each frame spans ``rows_per_frame`` data rows. The key
    sentinel (65535) lives in column ``cols`` and marks the *first* row of each
    frame. The next frame key appears ``rows_per_frame`` rows later, and that
    key row is also the last data row of the current frame.
    """
    # cols=2, nreps=3, rows=2 -> rows_per_frame = 6, raw_row_size = 3
    cols, nreps, rows, key_ints = 2, 3, 2, 1
    rows_per_frame = cols * nreps  # 6
    bin_file = tmp_path / "test.bin"

    data_rows: list[list[int]] = []
    for frame in range(2):
        for i in range(rows_per_frame):
            # The first row of each frame block is the key row.
            if i == 0:
                data_rows.append([0, 0, 65535])
            else:
                rep = (i - 1) // rows
                row = (i - 1) % rows
                data_rows.append([rep, row, frame * 10 + row])
    # Add the final key row that terminates the last frame.
    data_rows.append([0, 0, 65535])

    arr = np.array(data_rows, dtype="uint16")
    arr.tofile(bin_file)

    starts, ends, n_frames = _parse_bin_frames(
        bin_file, offset=0, nreps=nreps, cols=cols, rows=rows, key_ints=key_ints
    )
    assert n_frames == 2
    assert len(starts) == 2
    assert len(ends) == 2
    assert list(ends - starts) == [rows_per_frame, rows_per_frame]
