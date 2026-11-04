"""Tests for apantias.runtime.cluster."""

from pathlib import Path

import pytest

from apantias.runtime.cluster import init, resolve_worker_memory


def test_resolve_worker_memory_no_reduction() -> None:
    """When RAM is plentiful the requested core count is preserved."""
    cores, per_worker = resolve_worker_memory(
        cores=4,
        ram_mb=65536,
        min_worker_memory_mb=4096,
    )
    assert cores == 4
    assert per_worker == 14745  # 65536 * 0.9 / 4


def test_resolve_worker_memory_reduces_workers() -> None:
    """Too many cores for the RAM -> reduce workers to meet the memory floor."""
    cores, per_worker = resolve_worker_memory(
        cores=38,
        ram_mb=144400,
        min_worker_memory_mb=4096,
    )
    # 144400 * 0.9 / 38 == 3420, which is below the 4096 floor.
    # Available RAM allows floor, so workers are reduced.
    assert cores == 31
    assert per_worker >= 4096


def test_resolve_worker_memory_cannot_meet_minimum() -> None:
    """If total RAM is below the floor, keep the requested cores and cap at 256 MB."""
    cores, per_worker = resolve_worker_memory(
        cores=4,
        ram_mb=1024,
        min_worker_memory_mb=4096,
    )
    assert cores == 4
    assert per_worker == 256  # 1024 * 0.9 / 4 would be 230; capped at 256 MB


def test_resolve_worker_memory_disabled_floor() -> None:
    """A floor of 256 MB disables the reduction logic."""
    cores, per_worker = resolve_worker_memory(
        cores=38,
        ram_mb=144400,
        min_worker_memory_mb=256,
    )
    assert cores == 38
    assert per_worker == 3420


def test_init_fails_with_insufficient_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """cluster.init() refuses to start a cluster with fewer than 4 workers."""
    monkeypatch.setattr("apantias.runtime.resources.get_resources", lambda: (2, 4096))
    with pytest.raises(RuntimeError, match="At least 4 workers are required"):
        init(tmp_path / "dask_temp", dashboard=False, use_nanny=False)
