import logging
from pathlib import Path
import numpy as np
import zarr
from typing import cast
from . import utils

_logger = logging.getLogger(__name__)


class StandardAnalysis:
    """
    Performs standard statistical analysis on zarr arrays.

    Supports computing statistics (mean, std, etc.) over frames or spatial dimensions
    for multi-dimensional arrays with shape (frames, columns, repetitions, rows).
    """

    def __init__(self, bin_path: str | Path, temp_zarr: str | Path) -> None:
        """
        Initialize with a zarr store.

        Args:
            zarr_path: Path to the zarr store.
        """
        self.bin_path = Path(bin_path)
        self.temp_zarr = Path(temp_zarr)
        self.raw_data_pixelwise = self.temp_zarr.joinpath("raw_data", "pixel_chunked")
        self.raw_data_framewise = self.temp_zarr.joinpath("raw_data", "frame_chunked")

    def run(self):

        _logger.info("Start writing bin to zarr store.")

        # Check if raw_data_framewise already exists
        if self._array_exists(self.raw_data_framewise):
            _logger.info("raw_data_framewise already exists, skipping bin_to_zarr")
        else:
            _logger.info("Writing framewise data")
            utils.bin_to_zarr(self.bin_path, self.raw_data_framewise, 200)

        # Check if raw_data_pixelwise already exists
        if self._array_exists(self.raw_data_pixelwise):
            _logger.info("raw_data_pixelwise already exists, skipping rechunking")
        else:
            _logger.info("Start creating pixelwise chunked raw_data.")
            utils.rechunk_to_pixels(self.raw_data_framewise, self.raw_data_pixelwise)

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
