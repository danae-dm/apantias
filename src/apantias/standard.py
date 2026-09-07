import logging
import time
from collections.abc import Callable
from pathlib import Path

import dask.array as da
import zarr

from . import utils

_logger = logging.getLogger(__name__)


#
class StandardAnalysis:
    """
    Performs standard statistical analysis on zarr arrays.

    Supports computing statistics (mean, std, etc.) over frames or spatial dimensions
    for multi-dimensional arrays with shape (frames, columns, repetitions, rows).
    """

    def __init__(self, bin_path: str | Path, temp_zarr: str | Path, ext_offset: str | Path | None = None) -> None:
        """
        Initialize with a zarr store.

        Args:
            zarr_path: Path to the zarr store.
        """
        self.bin_path = Path(bin_path)
        self.temp_zarr = Path(temp_zarr)
        self.ext_offset = ext_offset
        self.raw_data_pixelwise = self.temp_zarr.joinpath("raw_data", "pixel_chunked")
        self.raw_data_framewise = self.temp_zarr.joinpath("raw_data", "frame_chunked")
        self.median = self.temp_zarr.joinpath("median")
        self.offset_corr = self.temp_zarr.joinpath("offset_corr")
        self.common_modes = self.temp_zarr.joinpath("common_modes")
        self.slopes = self.temp_zarr.joinpath("slopes")
        self.signals = self.temp_zarr.joinpath("signals")
        self.msd = self.temp_zarr.joinpath("msd")
        self.signals_mean = self.temp_zarr.joinpath("signals_mean")

    def _run_step(self, path: Path, step_name: str, func: Callable, *args, **kwargs) -> None:
        if self._array_exists(path):
            _logger.info("%s already exists, skipping", step_name)
        else:
            _logger.info("Start creating %s", step_name)
            start_time = time.perf_counter()
            func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            _logger.info("Finished %s in %.2fs", step_name, elapsed_time)

    def run(self):

        _logger.info("Start writing bin to zarr store.")

        self._run_step(
            self.raw_data_framewise,
            "raw_data_framewise",
            utils.bin_to_zarr,
            self.bin_path,
            self.raw_data_framewise,
            400,
        )

        self._run_step(
            self.raw_data_pixelwise,
            "raw_data_pixelwise",
            utils.rechunk_to_pixels,
            self.raw_data_framewise,
            self.raw_data_pixelwise,
        )

        data_p = da.from_zarr(self.raw_data_pixelwise)

        # Load frame-chunked data. Dask defaults to the inner chunk shape (e.g. col=1).
        # We explicitly rechunk axis 1 to full width (64) here so that offset_corr
        # and downstream steps process full frames and don't fragment into tiny tasks.
        # This does not affect rechunk_to_pixels, which runs before this and reads
        # directly from the zarr store.
        data_f = da.from_zarr(self.raw_data_framewise).rechunk({1: 64})

        self._run_step(self.median, "median", utils.compute_median, data_p, self.median)
        median = da.from_zarr(self.median)

        if self.ext_offset is None:
            offset = median
        else:
            offset = da.from_zarr(self.ext_offset)

        self._run_step(self.offset_corr, "offset_corr", utils.compute_offset_corr, data_f, offset, self.offset_corr)
        offset_corr = da.from_zarr(self.offset_corr)

        self._run_step(self.common_modes, "common_modes", utils.compute_common_modes, offset_corr, self.common_modes)
        common_modes = da.from_zarr(self.common_modes)

        self._run_step(self.signals, "signals", utils.subtract, offset_corr, common_modes, self.signals)
        signals = da.from_zarr(self.signals)

        self._run_step(self.slopes, "slopes", utils.compute_slopes, signals, self.slopes)

        self._run_step(self.msd, "mean squared deviation", utils.compute_msd, signals, median, self.msd)

        self._run_step(self.signals_mean, "signals_mean", utils.compute_signals_mean, signals, self.signals_mean)

    def _array_exists(self, path: Path) -> bool:
        """Check if a zarr array exists at the given path."""
        path_str = str(path)
        if ".zarr" in path_str:
            try:
                store_end = path_str.find(".zarr") + len(".zarr")
                store = path_str[:store_end]
                array_path = path_str[store_end:].lstrip("/")
                zarr.open_array(store, path=array_path, mode="r")  # type: ignore
                return True
            except (ValueError, FileNotFoundError, KeyError):
                return False
        else:
            try:
                zarr.open_array(path_str, mode="r")
                return True
            except (ValueError, FileNotFoundError, KeyError):
                return False
