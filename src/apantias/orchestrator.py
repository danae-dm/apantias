import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import dask
import dask.config
import zarr
import dask.array as da
from dask.distributed import Client, LocalCluster
import time
import numpy as np
import logging
import multiprocessing

from . import config as cf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run():
    if multiprocessing.parent_process() is not None:
        return  # we're in a spawned worker, bail out
    input_h5 = "raw_data_chunked_50MB.h5"
    zarr_store = "results.zarr"

    copy_raw_from_h5 = (
        False  # set False after first run if raw_data already exists in zarr
    )
    dask.config.set({"distributed.worker.multiprocessing-method": "forkserver"})
    cluster = LocalCluster(
        n_workers=6,
        threads_per_worker=1,
        processes=True,
        memory_limit="1GiB",
        local_directory="/tmp/dask-spill",
        dashboard_address=":8787",
    )
    client = Client(cluster)
    log.info(client.dashboard_link)
    log.info("Loading Config.")
    config = cf.load_config()
    print(config)

    # optional one-time copy from HDF5 -> Zarr
    if copy_raw_from_h5:
        with h5py.File(input_h5, "r") as f:
            x_h5 = da.from_array(f["raw_data"], chunks=f["raw_data"].chunks)
            x_h5 = x_h5[:-4]
            big = da.concatenate([x_h5] * 4, axis=0)
            print("start copying")
            da.to_zarr(
                big,
                zarr_store,
                component="raw_data",
                overwrite=True,
            )

        print("Copied /raw_data from HDF5 to Zarr and appended 15 copies")
    # load rawdata from the store, this is not in memory!
    x = da.from_zarr(zarr_store, component="raw_data")
    show_chunks("bla", x)
    # lazy computation of the mean, persist makes it stay in the workers ram
    offset = da.nanmean(x, axis=0, dtype="float32").persist()
    log.info("Compute and write Offset")
    # writing offset to zarr triggers the computation
    da.to_zarr(offset, zarr_store, component="raw_offset", overwrite=True)
    # the offset (still in the workers ram) is subtracted
    x_corr = x.astype("float32") - offset[None, :, :, :]
    # the median is defined, it should persist too, its subtracted later
    median = da.median(x_corr, axis=3).persist()
    log.info("Compute median.")
    # write to zarr (triggers computation)
    da.to_zarr(
        median,
        zarr_store,
        component="median",
        overwrite=True,
    )
    # define corrected
    log.info("Subtract median.")
    x_corr = x_corr - median[:, :, :, None]
    da.to_zarr(x_corr, zarr_store, component="signals", overwrite=True)
    log.info("Done")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
