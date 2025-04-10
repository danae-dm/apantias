"""module description"""

from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np
from numba import njit, prange
from scipy import stats
from sklearn.cluster import DBSCAN

from . import fitting


def shapiro(data: np.ndarray, axis: int):
    """
    Performs the Shapiro-Wilk test for normality along a specified axis of a NumPy array.

    Args:
        data (np.ndarray): Input data array.
        axis (int): Axis along which to perform the Shapiro-Wilk test.

    Returns:
        np.ndarray: An array of p-values indicating the likelihood that each slice follows a normal distribution.
    """
    result_shape = list(data.shape)
    result_shape.pop(axis)
    result = np.zeros(result_shape, dtype=float)

    # Iterate over the slices along the specified axis
    it = np.nditer(result, flags=["multi_index"])
    while not it.finished:
        # Get the slice indices
        idx = list(it.multi_index)
        idx.insert(int(axis), slice(None))  # type: ignore

        # Perform the Shapiro-Wilk test
        _, shapiro_p_value = stats.shapiro(data[tuple(idx)])

        # Determine if the slice follows a normal distribution
        result[it.multi_index] = shapiro_p_value

        it.iternext()

    return result


def get_avg_over_nreps(data: np.ndarray) -> np.ndarray:
    """
    Calculates the average over the nreps axis in a 4D array.

    Args:
        data (np.ndarray): Input data with shape (nframes, column_size, nreps, row_size).

    Returns:
        np.ndarray: Averaged data with shape (nframes, column_size, row_size).
    """
    if np.ndim(data) != 4:
        raise ValueError("Input data is not a 4D array.")
    return nanmean(data, axis=2)


def get_rolling_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculates a rolling average over a specified window size for 1D data.

    Args:
        data (np.ndarray): Input 1D data array.
        window_size (int): Size of the rolling window.

    Returns:
        np.ndarray: 1D array of rolling averages.
    """
    weights = np.repeat(1.0, window_size) / window_size
    # Use 'valid' mode to ensure that output has the same length as input
    return np.convolve(data, weights, mode="valid")


def get_ram_usage_in_gb(
    frames: int, column_size: int, nreps: int, row_size: int
) -> int:
    """
    Calculates the estimated RAM usage in GB for a 4D array with the given dimensions.

    Args:
        frames (int): Number of frames.
        column_size (int): Number of columns.
        nreps (int): Number of repetitions.
        row_size (int): Number of rows.

    Returns:
        int: Estimated RAM usage in gigabytes.
    """
    return int(frames * column_size * nreps * row_size * 8 / 1024**3) + 1


@njit(parallel=True)
def apply_slope_fit_along_frames(data):
    """
    Applies a slope fit along the nreps axis of a 4D array in parallel.

    Args:
        data (np.ndarray): Input 4D array with shape (nframes, column_size, nreps, row_size).

    Returns:
        np.ndarray: 3D array with shape (nframes, column_size, row_size) containing slope values.
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_3_size = data.shape[3]
    output = np.empty((axis_0_size, axis_1_size, axis_3_size))
    for frame in prange(axis_0_size):
        for row in range(axis_1_size):
            for col in range(axis_3_size):
                slope = fitting.linear_fit(data[frame, row, :, col])
                output[frame][row][col] = slope
    return output


@njit(parallel=False)
def apply_slope_fit_along_frames_single(data):
    """
    Applies a slope fit along the nreps axis of a 4D array (single-threaded).

    Args:
        data (np.ndarray): Input 4D array with shape (nframes, column_size, nreps, row_size).

    Returns:
        np.ndarray: 3D array with shape (nframes, column_size, row_size) containing slope values.
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_3_size = data.shape[3]
    output = np.empty((axis_0_size, axis_1_size, axis_3_size))
    for frame in prange(axis_0_size):
        for row in range(axis_1_size):
            for col in range(axis_3_size):
                slope = fitting.linear_fit(data[frame, row, :, col])
                output[frame][row][col] = slope
    return output


def split_h5_path(path: str) -> tuple[str, str]:
    """
    Splits an HDF5 file path into the file path and dataset path.

    Args:
        path (str): Full path to the HDF5 file and dataset (e.g., "/path/to/file.h5/group1/dataset1").

    Returns:
        tuple: A tuple containing the HDF5 file path and the dataset path.
    """
    h5_file = path.split(".h5")[0] + ".h5"
    dataset_path = path.split(".h5")[1]
    return h5_file, dataset_path


def nanmedian(data: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """
    Computes the median along a specified axis of a NumPy array, ignoring NaN values.

    Args:
        data (np.ndarray): Input data array.
        axis (int): Axis along which to compute the median.
        keepdims (bool): Whether to keep the reduced dimensions.

    Returns:
        np.ndarray: Array of medians along the specified axis.
    """
    if data.ndim == 2:
        if axis == 0:
            if keepdims:
                return _nanmedian_2d_axis0(data)[np.newaxis, :]
            else:
                return _nanmedian_2d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmedian_2d_axis1(data)[:, np.newaxis]
            else:
                return _nanmedian_2d_axis1(data)
    elif data.ndim == 3:
        if axis == 0:
            if keepdims:
                return _nanmedian_3d_axis0(data)[np.newaxis, :, :]
            else:
                return _nanmedian_3d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmedian_3d_axis1(data)[:, np.newaxis, :]
            else:
                return _nanmedian_3d_axis1(data)
        elif axis == 2:
            if keepdims:
                return _nanmedian_3d_axis2(data)[:, :, np.newaxis]
            else:
                return _nanmedian_3d_axis2(data)
    elif data.ndim == 4:
        if axis == 0:
            if keepdims:
                return _nanmedian_4d_axis0(data)[np.newaxis, :, :, :]
            else:
                return _nanmedian_4d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmedian_4d_axis1(data)[:, np.newaxis, :, :]
            else:
                return _nanmedian_4d_axis1(data)
        elif axis == 2:
            if keepdims:
                return _nanmedian_4d_axis2(data)[:, :, np.newaxis, :]
            else:
                return _nanmedian_4d_axis2(data)
        elif axis == 3:
            if keepdims:
                return _nanmedian_4d_axis3(data)[:, :, :, np.newaxis]
            else:
                return _nanmedian_4d_axis3(data)
    else:
        raise ValueError("Data has wrong dimensions")
    return np.array([])  # Add a default return statement


def nanmean(data: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """
    Computes the mean along a specified axis of a NumPy array, ignoring NaN values.

    Args:
        data (np.ndarray): Input data array.
        axis (int): Axis along which to compute the mean.
        keepdims (bool): Whether to keep the reduced dimensions.

    Returns:
        np.ndarray: Array of means along the specified axis.
    """
    if data.ndim == 2:
        if axis == 0:
            if keepdims:
                return _nanmean_2d_axis0(data)[np.newaxis, :]
            else:
                return _nanmean_2d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmean_2d_axis1(data)[:, np.newaxis]
            else:
                return _nanmean_2d_axis1(data)
    elif data.ndim == 3:
        if axis == 0:
            if keepdims:
                return _nanmean_3d_axis0(data)[np.newaxis, :, :]
            else:
                return _nanmean_3d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmean_3d_axis1(data)[:, np.newaxis, :]
            else:
                return _nanmean_3d_axis1(data)
        elif axis == 2:
            if keepdims:
                return _nanmean_3d_axis2(data)[:, :, np.newaxis]
            else:
                return _nanmean_3d_axis2(data)
    elif data.ndim == 4:
        if axis == 0:
            if keepdims:
                return _nanmean_4d_axis0(data)[np.newaxis, :, :, :]
            else:
                return _nanmean_4d_axis0(data)
        elif axis == 1:
            if keepdims:
                return _nanmean_4d_axis1(data)[:, np.newaxis, :, :]
            else:
                return _nanmean_4d_axis1(data)
        elif axis == 2:
            if keepdims:
                return _nanmean_4d_axis2(data)[:, :, np.newaxis, :]
            else:
                return _nanmean_4d_axis2(data)
        elif axis == 3:
            if keepdims:
                return _nanmean_4d_axis3(data)[:, :, :, np.newaxis]
            else:
                return _nanmean_4d_axis3(data)
    else:
        raise ValueError("Data has wrong dimensions")
    return np.array([])  # Add a default return statement


@njit(parallel=True)
def _nanmedian_4d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_1_size, axis_2_size, axis_3_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[:, i, j, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmedian_4d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size, axis_2_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[i, :, j, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmedian_4d_axis2(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=2, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size, axis_1_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[i, j, :, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmedian_4d_axis3(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=3, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_1_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_2_size):
                median = np.nanmedian(data[i, j, k, :])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmedian_3d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        raise ValueError("Input data is not a 3D array.")
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_1_size, axis_2_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            median = np.nanmedian(data[:, i, j])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmedian_3d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        raise ValueError("Input data is not a 3D array.")
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            median = np.nanmedian(data[i, :, j])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmedian_3d_axis2(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=2, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        raise ValueError("Input data is not a 3D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    output = np.zeros((axis_0_size, axis_1_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            median = np.nanmedian(data[i, j, :])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmedian_2d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    """
    if data.ndim != 2:
        raise ValueError("Input data is not a 2D array.")
    axis_1_size = data.shape[1]
    output = np.zeros(axis_1_size)
    for i in prange(axis_1_size):
        median = np.nanmedian(data[:, i])
        output[i] = median
    return output


@njit(parallel=True)
def _nanmedian_2d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    """
    if data.ndim != 2:
        raise ValueError("Input data is not a 2D array.")
    axis_0_size = data.shape[0]
    output = np.zeros(axis_0_size)
    for i in prange(axis_0_size):
        median = np.nanmedian(data[i, :])
        output[i] = median
    return output


@njit(parallel=True)
def _nanmean_4d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_1_size, axis_2_size, axis_3_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[:, i, j, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmean_4d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size, axis_2_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[i, :, j, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmean_4d_axis2(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=2, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size, axis_1_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[i, j, :, k])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmean_4d_axis3(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=3, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_1_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_2_size):
                median = np.nanmean(data[i, j, k, :])
                output[i, j, k] = median
    return output


@njit(parallel=True)
def _nanmean_3d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        raise ValueError("Input data is not a 3D array.")
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_1_size, axis_2_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            median = np.nanmean(data[:, i, j])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmean_3d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        raise ValueError("Input data is not a 3D array.")
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            median = np.nanmean(data[i, :, j])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmean_3d_axis2(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=2, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    """
    if data.ndim != 3:
        raise ValueError("Input data is not a 3D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    output = np.zeros((axis_0_size, axis_1_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            median = np.nanmean(data[i, j, :])
            output[i, j] = median
    return output


@njit(parallel=True)
def _nanmean_2d_axis0(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    """
    if data.ndim != 2:
        raise ValueError("Input data is not a 2D array.")
    axis_1_size = data.shape[1]
    output = np.zeros(axis_1_size)
    for i in prange(axis_1_size):
        median = np.nanmean(data[:, i])
        output[i] = median
    return output


@njit(parallel=True)
def _nanmean_2d_axis1(data: np.ndarray) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    """
    if data.ndim != 2:
        raise ValueError("Input data is not a 2D array.")
    axis_0_size = data.shape[0]
    output = np.zeros(axis_0_size)
    for i in prange(axis_0_size):
        median = np.nanmean(data[i, :])
        output[i] = median
    return output


@njit
def _find_root(label, parent):
    while parent[label] != label:
        label = parent[label]
    return label


@njit
def _union(label1, label2, parent):
    root1 = _find_root(label1, parent)
    root2 = _find_root(label2, parent)
    if root1 != root2:
        parent[root2] = root1


@njit
def two_pass_labeling(data, structure):
    """
    Implements the two-pass labeling algorithm for connected components in a 2D boolean array.

    Args:
        data (np.ndarray): 2D boolean array.
        structure (np.ndarray): Structuring element for connectivity.

    Returns:
        tuple: A tuple containing the labeled array and the number of features.
    """
    rows, cols = data.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    parent = np.arange(rows * cols, dtype=np.int32)
    next_label = 1

    # First pass
    for i in range(rows):
        for j in range(cols):
            if data[i, j] == 0:
                continue

            neighbors = []
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if structure[di + 1, dj + 1] == 1 and labels[ni, nj] != 0:
                            neighbors.append(labels[ni, nj])

            if not neighbors:
                labels[i, j] = next_label
                next_label += 1
            else:
                min_label = min(neighbors)
                labels[i, j] = min_label
                for neighbor in neighbors:
                    if neighbor != min_label:
                        _union(min_label, neighbor, parent)

    # Second pass
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:
                labels[i, j] = _find_root(labels[i, j], parent)

    # Relabel to ensure labels are contiguous
    unique_labels = np.unique(labels)
    label_map = np.zeros(unique_labels.max() + 1, dtype=np.int32)
    for new_label, old_label in enumerate(unique_labels):
        label_map[old_label] = new_label
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:
                labels[i, j] = label_map[labels[i, j]]
    num_features = (
        len(unique_labels) - 1
    )  # Subtract 1 to exclude the background label (0)

    return labels, num_features


def parse_numpy_slicing(slicing_str: str) -> list:
    """
    Parses a NumPy slicing string and converts it to a list of Python slice objects.

    Args:
        slicing_str (str): String representing NumPy slicing (e.g., "1:5, :, 2:10:2").

    Returns:
        list: List of Python slice objects.
    """
    slicing_str = slicing_str.replace("[", "")
    slicing_str = slicing_str.replace("]", "")
    slices = []
    slicing_parts = slicing_str.split(",")

    for part in slicing_parts:
        part = part.strip()
        if ":" in part:
            slice_parts = part.split(":")
            start = int(slice_parts[0]) if slice_parts[0] else None
            stop = int(slice_parts[1]) if slice_parts[1] else None
            step = (
                int(slice_parts[2]) if len(slice_parts) > 2 and slice_parts[2] else None
            )
            slices.append(slice(start, stop, step))
        else:
            slices.append(int(part))
    return slices


def process_batch(func, row_data, *args, **kwargs):
    """
    Applies a function to a row of data.

    Args:
        func (callable): Function to apply.
        row_data (np.ndarray): Row of data to process.
        *args: Additional arguments for the function.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        np.ndarray: Processed data.
    """

    def func_with_args(data):
        return func(data, *args, **kwargs)

    batch_results = np.apply_along_axis(func_with_args, axis=0, arr=row_data)
    return batch_results


def apply_pixelwise(cores, data, func, *args, **kwargs) -> np.ndarray:
    """
    Helper function to apply a function to each pixel in a 3D numpy array in parallel.
    Data must have shape (n,row,col). The function is applied to [:,row,col].
    A process is created for each row, to avoid overhead from creating too many processes.
    The passed function must accept a 1D array as input and must have a 1D array as output.
    The passed function must have a data parameter, which is the first argument.

    Args:
        cores (int): Number of CPU cores to use.
        data (np.ndarray): Input 3D array with shape (n, row, col).
        func (callable): Function to apply to each pixel.
        *args: Additional arguments for the function.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        np.ndarray: Processed data.
    """
    if data.ndim != 3:
        raise ValueError("Data must be a 3D array.")
    # try the passed function and check return value
    try:
        result = func(data[:, 0, 0], *args, **kwargs)
        result_shape = result.shape
        result_type = result.dtype
    except Exception as e:
        raise ValueError(f"Error applying function to data: {e}") from e
    if not isinstance(result, np.ndarray):
        raise ValueError("Function must return a numpy array.")
    if result.ndim != 1:
        raise ValueError("Function must return a 1D numpy array.")
    # initialize results, now that we know what the function returns
    if cores == 1:
        return func(data, *args, **kwargs)

    rows_per_process = divide_evenly(data.shape[1], cores)
    results = np.zeros(
        (result_shape[0], data.shape[1], data.shape[2]), dtype=result_type
    )
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(cores):
            # copy the data of one row and submit it to the executor
            # this is necessary to avoid memory issues
            process_data = data[
                :, sum(rows_per_process[:i]) : sum(rows_per_process[: i + 1]), :
            ]
            futures.append(
                executor.submit(
                    process_batch, func, process_data.copy(), *args, **kwargs
                )
            )
        # wait for all futures to be done
        wait(futures)
        # Process the results in the order they were submitted
        for i, future in enumerate(futures):
            try:
                batch_results = future.result()
                results[
                    :, sum(rows_per_process[:i]) : sum(rows_per_process[: i + 1]), :
                ] = batch_results
            except Exception as e:
                raise e
    return results


def dbscan_outliers(data: np.ndarray, eps, min_samples, inline=False):
    """
    Identifies outliers in data using the DBSCAN clustering algorithm.

    Args:
        data (np.ndarray): Input data array.
        eps (float): Maximum distance between two samples for them to be considered in the same cluster.
        min_samples (int): Minimum number of samples in a cluster.
        inline (bool): Whether to modify the input data inline.

    Returns:
        np.ndarray or None: Boolean mask of outliers if inline is False, otherwise modifies the input data.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data.reshape(-1, 1))
    if labels.shape != data.shape:
        labels = labels.reshape(data.shape)
    if not inline:
        return labels == -1
    else:
        data[labels == -1] = np.nan


def divide_evenly(number: int, parts: int) -> list:
    """
    Divides an integer into approximately equal parts.

    Args:
        number (int): Number to divide.
        parts (int): Number of parts to divide into.

    Returns:
        list: List of integers representing the divided parts.
    """
    quotient, remainder = divmod(number, parts)
    result = [quotient] * parts
    for i in range(remainder):
        result[i] += 1
    return result
