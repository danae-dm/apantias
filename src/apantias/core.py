import logging
import os

import dask.config
from dask.distributed import Client, LocalCluster

from apantias.get_resources import get_resources

from .utils import get_node_name

_logger = logging.getLogger(__name__)

# these environment variables ensure that numpy runs single threaded. we want only one thread running per core.
env_vars = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLOSC_NTHREADS": "1",
}


def init(local_dir, cores=None):
    # initialize the config
    if cores is None:
        cores = get_resources()[0]
    else:
        cores = max(1, cores - 1)
    local_directory = local_dir

    # Route the Dask dashboard through JupyterHub's server proxy
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    _ = dask.config.set({"distributed.dashboard.link": prefix + "proxy/{port}/status"})

    cluster = LocalCluster(
        n_workers=cores,
        threads_per_worker=1,
        env=env_vars,
        processes=True,
        dashboard_address=":8787",  # binds 0.0.0.0:8787 so the proxy can reach it
        local_directory=local_directory,
    )
    node = get_node_name()
    _logger.info(f"Initialized Dask on {cores} cores.")
    _logger.info("For Dashboard access open a ssh tunnel with this command:")
    _logger.info(f"ssh -N -L 8787:{node}:8787 user@cbe.vbc.ac.at")
    _logger.info("Access the Dashboard in your browser at http://localhost:8787")
    return Client(cluster), cluster
