"""Tests for apantias.inspect HDF5 printing helpers."""

# Tests exercise h5py's runtime API, which has incomplete stubs in strict mode.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from pathlib import Path

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from apantias.inspect.h5 import (
    print_archive_summary,
    print_dataset_info,
    print_dataset_stats,
    print_h5_contents,
)


@pytest.fixture
def sample_h5(tmp_path: Path) -> Path:
    h5_file = tmp_path / "sample.h5"
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("signals_mean", data=np.arange(12, dtype="float64").reshape(3, 4))
        f.create_dataset("pixelwise", data=np.ones((5, 5), dtype="float32"))
        group = f.create_group("nested")
        group.create_dataset("extra", data=np.zeros((2, 3)))
    return h5_file


def test_print_h5_contents(sample_h5: Path, capsys: pytest.CaptureFixture[str]) -> None:
    print_h5_contents(sample_h5)
    captured = capsys.readouterr().out
    assert str(sample_h5) in captured
    assert "signals_mean" in captured
    assert "pixelwise" in captured
    assert "nested/" in captured
    assert "extra" in captured


def test_print_archive_summary(sample_h5: Path, capsys: pytest.CaptureFixture[str]) -> None:
    print_archive_summary(sample_h5)
    captured = capsys.readouterr().out
    assert "Archive:" in captured
    assert "signals_mean" in captured
    assert "pixelwise" in captured
    assert "nested/" not in captured  # summary is flat top-level only


def test_print_dataset_info(sample_h5: Path, capsys: pytest.CaptureFixture[str]) -> None:
    print_dataset_info(sample_h5, "signals_mean")
    captured = capsys.readouterr().out
    assert "signals_mean" in captured
    assert "shape:" in captured
    assert "dtype:" in captured


def test_print_dataset_info_non_dataset(sample_h5: Path) -> None:
    with pytest.raises(ValueError, match="not a dataset"):
        print_dataset_info(sample_h5, "nested")


def test_print_dataset_stats(sample_h5: Path, capsys: pytest.CaptureFixture[str]) -> None:
    print_dataset_stats(sample_h5, "signals_mean")
    captured = capsys.readouterr().out
    assert "min:" in captured
    assert "max:" in captured
    assert "mean:" in captured
    assert "std:" in captured


def test_print_dataset_stats_sampling(sample_h5: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Force sampling by using a tiny max element limit.
    data = np.arange(100, dtype="float64")
    with h5py.File(sample_h5, "w") as f:
        f.create_dataset("large", data=data)

    print_dataset_stats(sample_h5, "large", sample_max_elements=10)
    captured = capsys.readouterr().out
    assert "sampled" in captured
    assert "mean:" in captured


def test_print_dataset_stats_non_dataset(sample_h5: Path) -> None:
    with pytest.raises(ValueError, match="not a dataset"):
        print_dataset_stats(sample_h5, "nested")
