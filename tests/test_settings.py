"""Tests for apantias.settings."""

from pathlib import Path

import yaml

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


def test_load_config_creates_default(tmp_path: Path, monkeypatch) -> None:
    # Avoid hanging on resource auto-detection in restricted environments.
    monkeypatch.setattr("apantias.get_resources.get_resources", lambda: (4, 8192))
    config_path = tmp_path / "default.yaml"
    assert not config_path.exists()
    cfg = load_config(config_path, quiet=True)
    assert config_path.exists()
    data = yaml.safe_load(config_path.read_text())
    assert "runtime" in data
    assert "frame" in data
    assert "analysis" in data
    assert cfg.runtime.cpus == 4
