import logging
from pathlib import Path

import dask.array as da
import zarr

from apantias.core import init
from apantias.settings import load_config

from . import utils

_logger = logging.getLogger(__name__)


class StandardAnalysis:
    """
    Performs standard statistical analysis on zarr arrays.

    Supports computing statistics (mean, std, etc.) over frames or spatial dimensions
    for multi-dimensional arrays with shape (frames, columns, repetitions, rows).
    """

    # config can be passed in, otherwise defaults to the shared frozen instance
    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            config = load_config()
        else:
            config = load_config(Path(path))
        self._client, self._cluster = init(config.runtime.dask_temp, config.runtime.cpus)
        self.config = config
        self.bin_path = Path(config.analysis.bin_file)
        self.zarr_data = Path(self.config.analysis.zarr_data)
        self.temp_zarr = Path(self.config.analysis.zarr_temp)
        self.raw_ext = self.config.analysis.ext_offset
        if self.raw_ext is not None and str(self.raw_ext) != "None":
            self.ext_offset = Path(self.raw_ext)
        else:
            self.ext_offset = None
        self.raw_data_pixelwise = self.zarr_data.joinpath("pixel_chunked")
        self.raw_data_framewise = self.zarr_data.joinpath("frame_chunked")
        self.median = self.temp_zarr.joinpath("median")
        self.offset_corr = self.temp_zarr.joinpath("offset_corr")
        self.common_modes = self.temp_zarr.joinpath("common_modes")
        self.slopes = self.temp_zarr.joinpath("slopes")
        self.signals = self.temp_zarr.joinpath("signals")
        self.msd = self.temp_zarr.joinpath("msd")
        self.signals_mean = self.temp_zarr.joinpath("signals_mean")

    def run(self):
        client, _ = self._client, self._cluster
        try:
            self._run_analysis()
        finally:
            client.close()

    def _run_analysis(self):

        zarr.open_group(self.temp_zarr, mode="a")
        zarr.open_group(self.zarr_data, mode="a")

        _logger.info("Start writing bin to zarr store.")
        utils.bin_to_zarr(self.bin_path, self.raw_data_framewise, self.config.frame.nreps)
        utils.rechunk_to_pixels(self.raw_data_framewise, self.raw_data_pixelwise)
        _logger.info("Done.")
        # Load pixelwise data. This must be used for calculations along frames.
        # The pixelwise data is saved in chunks per pixel, not per frame.
        data_p = da.from_zarr(self.raw_data_pixelwise)
        # Load frame-chunked data. Dask defaults to the inner chunk shape (e.g. col=1).
        # We explicitly rechunk axis 1 to full width (64) here so that offset_corr
        # and downstream steps process full frames and don't fragment into tiny tasks.
        # This does not affect rechunk_to_pixels, which runs before this and reads
        # directly from the zarr store.
        data_f = da.from_zarr(self.raw_data_framewise).rechunk({1: 64})

        _logger.info("Start calculating offset.")
        utils.compute_median(data_p, self.median)
        # load the median as dask array
        median = da.from_zarr(self.median)
        _logger.info("Done.")

        if self.ext_offset is None:
            offset = median
        else:
            offset = da.from_zarr(self.ext_offset)

        _logger.info("Start applying offset.")
        utils.compute_offset_corr(data_f, offset, self.offset_corr)
        offset_corr = da.from_zarr(self.offset_corr)
        _logger.info("Done.")

        _logger.info("Start Common Mode Correction")
        utils.compute_common_modes(offset_corr, self.common_modes)
        common_modes = da.from_zarr(self.common_modes)
        utils.subtract(offset_corr, common_modes, self.signals)
        signals = da.from_zarr(self.signals)
        _logger.info("Done.")

        _logger.info("Start calculating slopes")
        utils.compute_slopes(signals, self.slopes)
        _logger.info("Done.")

        _logger.info("Start calculating mean squared deviation")
        utils.compute_msd(signals, median, self.msd)
        _logger.info("Done.")

        _logger.info("Start calculating mean signals")
        utils.compute_signals_mean(signals, self.signals_mean)
        _logger.info("Done.")
