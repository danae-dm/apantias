from dask.distributed import Client, LocalCluster
import psutil
import logging
import os
from typing import Any
from apantias.utils import get_node_name
from pathlib import Path

_logger = logging.getLogger(__name__)

env_vars = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLOSC_NTHREADS": "1",
}


def _parse_slurm_mem(mem_str: str) -> float | None:
    if not mem_str:
        return None
    mem_str = mem_str.strip().upper()
    try:
        if mem_str.endswith("G"):
            return float(mem_str[:-1])
        elif mem_str.endswith("M"):
            return float(mem_str[:-1]) / 1024.0
        elif mem_str.endswith("K"):
            return float(mem_str[:-1]) / (1024.0 * 1024.0)
        else:
            # SLURM default is usually MB
            return float(mem_str) / 1024.0
    except ValueError:
        return None


def _get_cgroup_memory_limit() -> float | None:
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    val = f.read().strip()
                if val and val != "max":
                    limit = int(val)
                    # Some systems return a very high number (e.g., 9223372036854771712) when no limit is set
                    if limit < 9000000000000000000:
                        return limit / (1024.0**3)
            except Exception:
                pass
    return None


def get_resources() -> dict[str, Any]:
    """
    Detect the resource allocations (CPUs, Memory, GPUs) available to the current process,
    respecting SLURM environment variables, container cgroup limits, and system fallbacks.
    """
    # 1. CPU allocation
    cpus = None
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        try:
            cpus = int(slurm_cpus)
        except ValueError:
            pass

    if cpus is None and hasattr(os, "sched_getaffinity"):
        try:
            cpus = len(os.sched_getaffinity(0))
        except Exception:
            pass

    if cpus is None or cpus <= 0:
        physical_cores = psutil.cpu_count(logical=False)
        cpus = physical_cores if physical_cores else (os.cpu_count() or 1)

    # 2. Memory allocation
    memory_gb = None
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem:
        memory_gb = _parse_slurm_mem(slurm_mem)

    if memory_gb is None:
        slurm_mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
        if slurm_mem_per_cpu:
            parsed_per_cpu = _parse_slurm_mem(slurm_mem_per_cpu)
            if parsed_per_cpu is not None:
                memory_gb = parsed_per_cpu * cpus

    if memory_gb is None:
        memory_gb = _get_cgroup_memory_limit()

    if memory_gb is None:
        memory_gb = psutil.virtual_memory().total / (1024.0**3)

    resources: dict[str, Any] = {"cpus": cpus, "memory_gb": round(memory_gb, 2)}
    _logger.info(f"Detected resources: {resources}")
    return resources


def init_cluster(local_directory: str | Path, cores: int = 0):
    resources = get_resources()
    detected_cores = resources["cpus"]

    if cores == 0:
        cores = max(1, detected_cores - 1)
    elif cores > detected_cores:
        _logger.warning(
            f"Requested {cores} cores, but only {detected_cores} are available. Limiting to {detected_cores}."
        )
        cores = detected_cores

    # Route the Dask dashboard through JupyterHub's server proxy
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    try:
        import dask.config  # type: ignore
    except ImportError:
        pass
    dask.config.set({"distributed.dashboard.link": prefix + "proxy/{port}/status"})  # type: ignore

    cluster = LocalCluster(
        n_workers=cores,
        threads_per_worker=1,
        env=env_vars,
        processes=True,
        dashboard_address=":8787",  # binds 0.0.0.0:8787 so the proxy can reach it
        local_directory=local_directory,
    )
    node = get_node_name()
    _logger.info("For Dashboard access open a ssh tunnel with this command:")
    _logger.info(f"ssh -N -L 8787:{node}:8787 user@cbe.vbc.ac.at")
    _logger.info("Access the Dashboard in your browser at http://localhost:8787")
    return Client(cluster)
