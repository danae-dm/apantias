import logging
from pathlib import Path

from apantias.settings import AppSettings, get_config

from . import utils

_logger = logging.getLogger(__name__)


class StandardAnalysis:
    """
    Performs standard statistical analysis on zarr arrays.

    Supports computing statistics (mean, std, etc.) over frames or spatial dimensions
    for multi-dimensional arrays with shape (frames, columns, repetitions, rows).
    """

    # config can be passed in, otherwise defaults to the shared frozen instance
    def __init__(self, config: AppSettings | None = None) -> None:
        if config is None:
            config = get_config()
        self.config = config
        self.bin_path = Path(config.analysis.bin_file)
        self.zarr_data = Path(self.config.analysis.zarr_data)
        self.temp_zarr = Path(self.config.analysis.zarr_temp)
        if self.config.analysis.ext_offset is not None:
            self.ext_offset = Path(self.config.analysis.ext_offset)
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

        _logger.info("Start writing bin to zarr store.")

        utils.bin_to_zarr(self.bin_path, self.raw_data_framewise, self.config.frame.nreps)

        utils.rechunk_to_pixels(self.raw_data_framewise, self.raw_data_pixelwise)

        _logger.info("Done.")
