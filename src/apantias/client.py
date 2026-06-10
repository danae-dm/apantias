from dask.distributed import Client, LocalCluster
import psutil
import logging
from urllib.parse import urlparse

_logger = logging.getLogger(__name__)

env_vars = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLOSC_NTHREADS": "1",
}


def init_cluster(cores: int = 0):

    physical_cores: int | None = psutil.cpu_count(logical=False)
    if physical_cores is None:
        _logger.error("No physical cores found.")
        return None

    if cores == 0:
        cores = physical_cores - 1
        _logger.info(f"Detected {physical_cores} physical cores. Dask will use {cores} cores.")
    elif cores > physical_cores:
        cores = physical_cores - 1
        _logger.warning(
            f"Requested {cores} cores, but only {physical_cores} physical cores are available. "
            f"Dask will use {cores} cores."
        )
    else:
        _logger.info(f"{cores} cores will be used of {physical_cores} available.")
    cluster = LocalCluster(
        n_workers=cores,
        threads_per_worker=1,
        env=env_vars,
        scheduler_port=8786,
        processes=True,
    )
    dashboard_url = cluster.dashboard_link
    parsed = urlparse(dashboard_url)
    port = parsed.port
    _logger.info(f"Dashboard Link: {dashboard_url}")
    _logger.info(f"If this runs in a jupyter container, try: http://localhost:8888/proxy/{port}/status")
    return Client(cluster)
