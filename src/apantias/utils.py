import gc
import os

import numpy as np
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor, wait
from scipy import stats
from sklearn.cluster import DBSCAN

from . import fitting


def shapiro(data, axis):
    result_shape = list(data.shape)
    result_shape.pop(axis)
    result = np.zeros(result_shape, dtype=float)

    # Iterate over the slices along the specified axis
    it = np.nditer(result, flags=["multi_index"])
    while not it.finished:
        # Get the slice indices
        idx = list(it.multi_index)
        idx.insert(axis, slice(None))

        # Perform the Shapiro-Wilk test
        shapiro_stat, shapiro_p_value = stats.shapiro(data[tuple(idx)])

        # Determine if the slice follows a normal distribution
        result[it.multi_index] = shapiro_p_value

        it.iternext()

    return result


def get_avg_over_nreps(data: np.ndarray) -> np.ndarray:
    """
    Calculates the average over the nreps in data.
    Args:
        data: in shape (nframes, column_size, nreps, row_size)
    Returns:
        np.array in shape (nframes, column_size, row_size)
    """
    if np.ndim(data) != 4:
        raise ValueError("Input data is not a 4D array.")
    return nanmean(data, axis=2)


def get_rolling_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculates a rolling average over window_size
    Args:
        data: in 1 dimension
        window_size:
    Returns:
        1D np.array
    """
    weights = np.repeat(1.0, window_size) / window_size
    # Use 'valid' mode to ensure that output has the same length as input
    return np.convolve(data, weights, mode="valid")


def get_ram_usage_in_gb(
    frames: int, column_size: int, nreps: int, row_size: int
) -> int:
    """
    Calculates the RAM usage in GB for a 4D array of the given dimensions.
    Assuming float64. (8 bytes per element)
    """
    return int(frames * column_size * nreps * row_size * 8 / 1024**3) + 1


@njit(parallel=True)
def apply_slope_fit_along_frames(data):
    """
    The equivalent to np.apply_along_axis(func, axis=2, data).
    Args:
        data: 4D np.array
    Returns:
        3D np.array (frame,row,col) with slope value
    """
    if data.ndim != 4:
        raise ValueError("Input data is not a 4D array.")
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.empty((axis_0_size, axis_1_size, axis_3_size))
    for frame in prange(axis_0_size):
        for row in range(axis_1_size):
            for col in range(axis_3_size):
                slope = fitting.linear_fit(data[frame, row, :, col])
                output[frame][row][col] = slope[0]
    return output


def split_h5_path(path: str) -> tuple:
    """
    Splits the h5_path into the directory and the filename.
    Example path: /path/to/file.h5/group1/dataset1
    Args:
        path: str
    Returns:
        tuple: (directory, filename)
    """
    h5_file = path.split(".h5")[0] + ".h5"
    dataset_path = path.split(".h5")[1]
    return h5_file, dataset_path


def nanmedian(data: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """
    The equivalent to np.nanmedian(data, axis=axis, keepdims=keepdims).
    Runs in parallel using numba.
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


def nanmean(data: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    """
    The equivalent to np.nanmean(data, axis=axis, keepdims=keepdims).
    Runs in parallel using numba.
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
    Implementation of the two pass labelling algortihm. The function take a 2d boolean array
    and groups the true values according to the structure element.
    0 in the output means no group, so the first group has index 1!
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
    Parses a NumPy array slicing string and converts it to a list of Python slice objects.

    Args:
        slicing_str: A string representing NumPy array slicing (e.g., "1:5, :, 2:10:2").

    Returns:
        A list of Python slice objects.
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
    Helper function to apply a function to a row of data.
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
    """
    if data.ndim != 3:
        raise ValueError("Data must be a 3D array.")
    # try the passed function and check return value
    try:
        result = func(data[:, 0, 0], *args, **kwargs)
        result_shape = result.shape
        result_type = result.dtype
    except Exception as e:
        raise ValueError(f"Error applying function to data: {e}")
    if not isinstance(result, np.ndarray):
        raise ValueError("Function must return a numpy array.")
    if result.ndim != 1:
        raise ValueError("Function must return a 1D numpy array.")
    # initialize results, now that we know what the function returns
    if cores == 1:
        return func(data, *args, **kwargs)

    rows_per_process = divide_evenly(data.shape[1], cores)
    results = np.zeros((result_shape[0], data.shape[1], data.shape[2]), dtype=result_type)
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
        #wait for all futures to be done
        done, not_done = wait(futures)
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


def dbscan_outliers(data, eps, min_samples, inline=False):
    """
    i tested this for some pixel histograms and it works well
    eps=1 and min_samples=2 seems to work well
    that means a pixel is considered an outlier if it is not part of a cluster of at least 2 pixels
    (including itself, min_samples=2) that are within a distance of 1 (eps) from each other
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data.reshape(-1, 1))
    if labels.shape != data.shape:
        labels = labels.reshape(data.shape)
    if not inline:
        return (labels == -1)
    else:
        data[labels == -1] = np.nan

def divide_evenly(number, parts):
    """
    Divides an integer number evenly into a set of integers.
    Args:
        number: The integer number to be divided.
        parts: The number of parts to divide into.
    Returns:
        A list of integers representing the divided parts.
    """
    quotient, remainder = divmod(number, parts)
    result = [quotient] * parts
    for i in range(remainder):
        result[i] += 1
    return result
