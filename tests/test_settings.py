"""Tests for apantias.settings."""

from pathlib import Path

import pytest
import yaml
from pytest import MonkeyPatch

from apantias.settings import AppSettings, RangeSpec, load_config


def test_rangespec_length() -> None:
    assert len(RangeSpec(start=0, stop=10, step=1)) == 10
    assert len(RangeSpec(start=0, stop=10, step=2)) == 5
    assert len(RangeSpec(start=5, stop=5, step=1)) == 0


def test_rangespec_iteration() -> None:
    spec = RangeSpec(start=0, stop=5, step=2)
    assert list(spec) == [0, 2, 4]


def test_appsettings_defaults() -> None:
    cfg = AppSettings()
    assert cfg.frame.rows == 64
    assert cfg.frame.cols == 64
    assert cfg.frame.nreps == 200


def test_load_config_creates_default(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Avoid hanging on resource auto-detection in restricted environments.
    monkeypatch.setattr("apantias.runtime.resources.get_resources", lambda: (4, 8192))
    config_path = tmp_path / "default.yaml"
    assert not config_path.exists()
    cfg = load_config(config_path, quiet=True)
    assert config_path.exists()
    data = yaml.safe_load(config_path.read_text())
    assert "runtime" in data
    assert "frame" in data
    assert "analysis" in data
    assert cfg.runtime.min_worker_memory_mb == 4096


def test_load_config_missing_field_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Loading a YAML with missing fields must raise ValueError (not silently default)."""
    monkeypatch.setattr("apantias.runtime.resources.get_resources", lambda: (4, 8192))

    yaml_content = """
runtime:
  dask_temp: /tmp/dask
  dask_dashboard: false
  dask_nanny: false
  min_worker_memory_mb: 2048
frame:
  target_chunk_mb: 150
analysis:
  h5_file: /tmp/data.h5
  zarr_temp: /tmp/temp.zarr
  keep_zarr_temp: false
  h5_archive: /tmp/archive.h5
  ext_offset: null
  nreps_range:
    start: 0
    stop: -1
    step: 1
  frames_range:
    start: 0
    stop: -1
    step: 1
  slope_threshold: 3
  second_offset: true
  # raw_sign_flip intentionally omitted
"""
    config_path = tmp_path / "incomplete.yaml"
    config_path.write_text(yaml_content)

    with pytest.raises(ValueError) as exc_info:
        load_config(config_path, quiet=True)

    assert "raw_sign_flip" in str(exc_info.value)
