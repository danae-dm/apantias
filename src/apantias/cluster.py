import logging
import os
from pathlib import Path

import dask.config
from dask.distributed import Client, LocalCluster

from apantias.get_resources import get_resources

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


def init(local_dir: str | Path, cores: int | None = None, ram_mb: int | None = None) -> tuple[Client, LocalCluster]:
    # initialize the config
    if cores is None:
        cores = get_resources()[0]
    else:
        cores = max(1, cores)
    local_directory = str(local_dir)

    # Per-worker memory limit from the configured total allocation. Distributed's
    # auto-detect sees the whole node, not the SLURM allocation, so an explicit
    # limit keeps the pause/kill thresholds inside what the job actually gets.
    # 10% headroom covers the driver process and scheduler/protocol overhead.
    memory_limit = None
    if ram_mb is not None and ram_mb > 0:
        per_worker_mb = max(256, int(ram_mb * 0.9 / max(1, cores)))
        memory_limit = f"{per_worker_mb}MB"

    # Route the Dask dashboard through JupyterHub's server proxy
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    _ = dask.config.set({"distributed.dashboard.link": prefix + "proxy/{port}/status"})

    cluster = LocalCluster(
        n_workers=cores,
        threads_per_worker=1,
        env=env_vars,
        processes=True,
        memory_limit=memory_limit,
        dashboard_address=":43359",  # binds 0.0.0.0:8787 so the proxy can reach it
        local_directory=local_directory,
    )
    node = get_node_name()
    _logger.info(f"Initialized Dask on {cores} cores.")
    if memory_limit is not None:
        _logger.info(f"Worker memory limit: {memory_limit} per worker.")
    _logger.info("For Dashboard access open a ssh tunnel with this command:")
    _logger.info(f"ssh -N -L 8787:{node}:43359 user@cbe.vbc.ac.at")
    _logger.info("Access the Dashboard in your browser at http://localhost:8787")
    return Client(cluster), cluster
