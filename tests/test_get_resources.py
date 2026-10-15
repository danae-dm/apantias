"""Tests for apantias.get_resources."""

from apantias.get_resources import _parse_slurm_mem, get_resources


def test_parse_slurm_mem() -> None:
    assert _parse_slurm_mem("1024") == 1024
    assert _parse_slurm_mem("2G") == 2048
    assert _parse_slurm_mem("512M") == 512
    assert _parse_slurm_mem("1024K") == 1
    assert _parse_slurm_mem("") is None
    assert _parse_slurm_mem("abc") is None


def test_get_resources_slurm(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "2048")
    cpus, mem = get_resources()
    assert cpus == 4
    assert mem == 2048


def test_get_resources_mem_per_cpu(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1024")
    for key in ("SLURM_MEM_PER_NODE",):
        monkeypatch.delenv(key, raising=False)
    cpus, mem = get_resources()
    assert cpus == 2
    assert mem == 2048
