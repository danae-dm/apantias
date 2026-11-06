import logging
import os
from pathlib import Path
from typing import Any

from dask.distributed import Client, LocalCluster
from distributed.worker import Worker

from apantias.runtime import resources

_logger = logging.getLogger(__name__)

# these environment variables ensure that numpy runs single threaded. we want only one thread running per core.
env_vars = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLOSC_NTHREADS": "1",
}


def get_node_name() -> str:
    """Get the name of the compute node the process is running on.

    Uses the SLURMD_NODENAME environment variable set by SLURM,
    or falls back to the hostname if not running on a SLURM cluster.

    Returns
    -------
    str
        The name of the node.
    """
    node_name = os.environ.get("SLURMD_NODENAME")
    if node_name is None:
        import socket

        node_name = socket.gethostname()
    return node_name


def resolve_worker_memory(
    cores: int,
    ram_mb: int,
    min_worker_memory_mb: int,
) -> tuple[int, int]:
    """Return ``(n_workers, per_worker_mb)`` honoring a per-worker memory floor.

    The calculation reserves 10% of ``ram_mb`` for scheduler/driver overhead,
    then divides the rest by ``cores``. If that share is below
    ``min_worker_memory_mb`` and total RAM is large enough to honor the floor,
    the worker count is reduced so each worker gets at least the minimum.
    """
    available_mb = ram_mb * 0.9
    per_worker_mb = max(256, int(available_mb / max(1, cores)))
    if min_worker_memory_mb > 256 and per_worker_mb < min_worker_memory_mb and available_mb >= min_worker_memory_mb:
        cores = max(1, int(available_mb / min_worker_memory_mb))
        per_worker_mb = max(min_worker_memory_mb, int(available_mb / max(1, cores)))
    return cores, int(per_worker_mb)


# Backward compatibility alias
_resolve_worker_memory = resolve_worker_memory


def init(
    local_dir: str | Path,
    *,
    min_worker_memory_mb: int = 4096,
    dashboard: bool = True,
    use_nanny: bool = True,
) -> tuple[Client, LocalCluster]:
    # Always use the resources actually available to this process.  Manual
    # core/RAM overrides are not supported because the pipeline is designed
    # to scale to the allocation while respecting the per-worker memory floor.
    detected_cores, total_ram_mb = resources.get_resources()
    local_directory = str(local_dir)

    # Make sure numpy and related libraries run single-threaded in the parent
    # process and in all worker subprocesses.
    os.environ.update(env_vars)

    # Per-worker memory limit from the configured total allocation. Distributed's
    # auto-detect sees the whole node, not the SLURM allocation, so an explicit
    # limit keeps the pause/kill thresholds inside what the job actually gets.
    # 10% headroom covers the driver process and scheduler/protocol overhead.
    # If the naive per-worker share is below the requested minimum (and total RAM
    # is large enough to honor it), reduce the worker count instead of running
    # many memory-starved workers.
    n_workers, per_worker_mb = _resolve_worker_memory(detected_cores, total_ram_mb, min_worker_memory_mb)
    if n_workers < 4:
        raise RuntimeError(
            f"Insufficient resources for Dask cluster: only {n_workers} worker(s) can be "
            f"started while guaranteeing {min_worker_memory_mb} MB RAM per worker "
            f"(available: {detected_cores} cores, {total_ram_mb} MB RAM). At least 4 workers are required."
        )
    _logger.info(
        "Dask worker memory: %d MB per worker on %d workers.",
        per_worker_mb,
        n_workers,
    )
    memory_limit = f"{per_worker_mb}MB"

    cluster_kwargs: dict[str, Any] = {
        "n_workers": n_workers,
        "threads_per_worker": 1,
        "processes": True,
        "memory_limit": memory_limit,
        "dashboard_address": "127.0.0.1:0" if dashboard else None,  # pyright: ignore[reportArgumentType]  # random free port on localhost
        "local_directory": local_directory,
    }
    if use_nanny:
        cluster_kwargs["env"] = env_vars
    else:
        cluster_kwargs["worker_class"] = Worker

    cluster = LocalCluster(**cluster_kwargs)
    _logger.info(f"Initialized Dask on {n_workers} workers.")
    _logger.info(f"Worker memory limit: {memory_limit} per worker.")
    if dashboard:
        port = cluster.scheduler.port
        _logger.info("Dask dashboard running on port %d (localhost only)", port)
        _logger.info("SSH tunnel: ssh -N -L 8787:%s:%d user@cbe.vbc.ac.at", get_node_name(), port)
        _logger.info("Access in browser: http://localhost:8787")
    else:
        _logger.info("No dashboard created.")
    return Client(cluster), cluster
