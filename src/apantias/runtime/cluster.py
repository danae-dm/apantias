import logging
import os
from pathlib import Path
from typing import Any

import dask.config
from dask.distributed import Client, LocalCluster
from distributed.worker import Worker

from apantias.runtime.resources import get_resources

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


def _resolve_worker_memory(
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


def init(
    local_dir: str | Path,
    cores: int | None = None,
    ram_mb: int | None = None,
    *,
    min_worker_memory_mb: int = 4096,
    dashboard: bool = True,
    use_nanny: bool = True,
) -> tuple[Client, LocalCluster]:
    # initialize the config
    if cores is None:
        cores = get_resources()[0]
    else:
        cores = max(1, cores)
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
    memory_limit = None
    if ram_mb is not None and ram_mb > 0:
        cores, per_worker_mb = _resolve_worker_memory(cores, ram_mb, min_worker_memory_mb)
        if per_worker_mb >= min_worker_memory_mb:
            _logger.info(
                "Dask worker memory: %d MB per worker on %d workers.",
                per_worker_mb,
                cores,
            )
        memory_limit = f"{per_worker_mb}MB"

    # Route the Dask dashboard through JupyterHub's server proxy
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    _ = dask.config.set({"distributed.dashboard.link": prefix + "proxy/{port}/status"})

    # When the dashboard is disabled, distributed still starts an HTTP server on
    # the default port (8787). Override the scheduler bind address to :0 so it
    # picks a random free port instead of warning about 8787 being in use.
    scheduler_kwargs = {"dashboard_address": ":0"} if not dashboard else None

    cluster_kwargs: dict[str, Any] = {
        "n_workers": cores,
        "threads_per_worker": 1,
        "processes": True,
        "memory_limit": memory_limit,
        "dashboard_address": ":43359" if dashboard else None,  # pyright: ignore[reportArgumentType]  # None disables dashboard
        "scheduler_kwargs": scheduler_kwargs,
        "local_directory": local_directory,
    }
    if use_nanny:
        cluster_kwargs["env"] = env_vars
    else:
        cluster_kwargs["worker_class"] = Worker

    cluster = LocalCluster(**cluster_kwargs)
    _logger.info(f"Initialized Dask on {cores} cores.")
    if memory_limit is not None:
        _logger.info(f"Worker memory limit: {memory_limit} per worker.")
    if dashboard:
        node = get_node_name()
        _logger.info("For Dashboard access open a ssh tunnel with this command:")
        _logger.info(f"ssh -N -L 8787:{node}:43359 user@cbe.vbc.ac.at")
        _logger.info("Access the Dashboard in your browser at http://localhost:8787")
    return Client(cluster), cluster
