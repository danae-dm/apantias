"""
This module provides various functions and utilities for data analysis,
including slope calculation, common mode correction, and pixel grouping
using threshold-based labeling. It leverages Numba for parallel processing
to optimize performance on large datasets.
"""

import numpy as np
from numba import njit, prange
from scipy import ndimage
import tables

from . import utils


def get_slopes(data: np.ndarray) -> np.ndarray:
    """
    Calculates the slope over nreps for every pixel and frame in parallel using numba.
    Args:
        data: (nframes, column_size, nreps, row_size)
    Returns:
        slopes: (nframes, column_size, row_size)
    """
    if np.ndim(data) != 4:
        raise ValueError("Input data is not a 4D array.")
    slopes = utils.apply_slope_fit_along_frames(data)
    return slopes


def correct_common_mode(data: np.ndarray) -> None:
    """
    Calculates the median of euch row in data, and substracts it from
    the row. The median is calculated in parallel using numba.
    Correction is done inline to save memory.
    Args:
        data: (nframes, column_size, nreps, row_size)
    """
    if data.ndim != 4:
        raise ValueError("Data is not a 4D array")
    median_common = utils.nanmedian(data, axis=3, keepdims=True)
    data -= median_common


def create_event_data(
    h5_file: str,
    path: str,
    table_name: str,
    data: np.ndarray,
    primary_threshold: float,
    secondary_threshold: float,
    noise_map: np.ndarray,
    structure: np.ndarray,
):
    """ """
    print("start creating event dict")
    event_data = utils.create_event_data(data, primary_threshold, secondary_threshold, noise_map, structure)
    print("start writing event dict")
    utils.write_event_data_to_h5(event_data, h5_file, path, table_name)


def query_event_data(h5_filename: str, query: str, path: str, table_name: str) -> np.ndarray:
    """
    Query event data from HDF5 file with conditions.

    Args:
        h5_filename: Path to the HDF5 file
        query: Query string (e.g., "is_primary == True")
        table_name: Name of the table in the HDF5 file (default: "events")

    Returns:
        Filtered numpy array with event data
    """
    try:
        with tables.open_file(h5_filename, mode="r") as h5file:
            table = h5file.get_node(path, table_name)
            if not isinstance(table, tables.Table):
                # Return empty array if not a Table
                return np.array([])

            # Create a query iterator for large datasets
            query_iter = table.where(query)

            # Collect results - for small datasets, read all
            filtered_data = np.array([row[:] for row in query_iter])

            # If no results, create empty array with correct dtype
            if len(filtered_data) == 0:
                filtered_data = np.array([], dtype=table.dtype)

            print(f"Query '{query}' returned {len(filtered_data)} events")

            return filtered_data
    except Exception as e:
        return np.array([])
    return np.array([])
