import gc
import os

import numpy as np

from . import parallel_funcs
from . import logger

_logger = logger.Logger(__name__, 'info').get_logger()

def get_avg_over_frames(data: np.ndarray) -> np.ndarray:
    '''
    Calculates the average over the frames in data.
    Args:
        data: np.array (nframes, column_size, nreps, row_size)
    
    Returns:
        np.array (column_size, nreps, row_size)
    '''
    if np.ndim(data) != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    return parallel_funcs.nanmean(data, axis=0)

def get_avg_over_nreps(data: np.ndarray) -> np.ndarray:
    '''
    Calculates the average over the nreps in data.
    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
    Returns:
        np.array in shape (nframes, column_size, row_size)
    '''
    if np.ndim(data) != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    return parallel_funcs.nanmean(data, axis=2)

def get_avg_over_frames_and_nreps(data : np.ndarray,
                                  avg_over_frames: np.ndarray = None,
                                  avg_over_nreps: np.ndarray = None) -> np.ndarray:
    '''
    Calculates the average over the frames and nreps in data. If avg_over_frames
    or avg_over_nreps are already calculated they can be passed as arguments to
    save computation time.
    Args:
        data: np.array (nframes, column_size, nreps, row_size)
        avg_over_frames: (optional) np.array (column_size, nreps, row_size)
        avg_over_nreps: (optional) np.array (nframes, column_size, row_size)
    Returns:
        np.array (column_size, row_size)
    '''
    if np.ndim(data) != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    
    if avg_over_frames is None and avg_over_nreps is None:
        return parallel_funcs.nanmean(parallel_funcs.nanmean(data, axis=0), axis = 2)
    
    if avg_over_frames is not None and avg_over_nreps is not None:
        if np.ndim(avg_over_frames) != 3 or np.ndim(avg_over_nreps) != 3:
            _logger.error('Input avg_over_frames or avg_over_nreps is not a 3D array.')
            raise ValueError('Input avg_over_frames or avg_over_nreps is not a 3D array.')
        if avg_over_frames.shape[1] < avg_over_nreps.shape[0]:
            return parallel_funcs.nanmean(avg_over_frames, axis=1)
        else:
            return parallel_funcs.nanmean(avg_over_nreps, axis=0)
    else: 
        if avg_over_nreps is not None:
            if np.ndim(avg_over_nreps) != 3:
                _logger.error('Input avg_over_nreps is not a 3D array.')
                raise ValueError('Input avg_over_nreps is not a 3D array.')
            return parallel_funcs.nanmean(avg_over_nreps, axis=0)
        else:
            if np.ndim(avg_over_frames) != 3:
                _logger.error('Input avg_over_frames is not a 3D array.')
                raise ValueError('Input avg_over_frames is not a 3D array.')
            return parallel_funcs.nanmean(avg_over_frames, axis=1)

def get_rolling_average(data: np.ndarray, window_size: int) -> np.ndarray:
    '''
    Calculates a rolling average over window_size
    Args:
        data: 1D np.array
        window_size: int
    Returns:
        1D np.array
    '''
    weights = np.repeat(1.0, window_size) / window_size
    # Use 'valid' mode to ensure that output has the same length as input
    return np.convolve(data, weights, mode='valid')

def load_npy_files(folder: str) -> dict:
    '''
    Looks for .npy arrays in folder and returns them all as a
    dictionary with numpy arrays as values
    Args:
        folder: folder path
    Returns:
        dictionary of np.arrays
    '''
    # Get a list of all .npy files in the folder
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if len(files) == 0:
        print(f'No .npy files found in folder {folder}')
        return None
    # Load each file into a numpy array and store it in a dictionary
    arrays = {}
    for file in files:
        # Remove the .npy extension from the filename
        name = os.path.splitext(file)[0]
        # Load the file and store it in the dictionary
        arrays[name] = np.load(os.path.join(folder, file), allow_pickle=True)
    return arrays

def sort_with_indices(arr: np.ndarray) -> np.ndarray:
    '''
    Sorts array in descending order and returns the indices.
    Args:
        arr: 1D np.array
    Returns:
        1D np.array
    '''
    if np.ndim(arr) != 1:
        _logger.error('Input data is not a 1D array.')
        raise ValueError('Input data is not a 1D array.')
    indexed_arr = np.column_stack((np.arange(len(arr)), arr))
    sorted_indices = indexed_arr[np.argsort(indexed_arr[:, 1])[::-1]][:, 0]
    return sorted_indices.astype(int)

def get_array_from_file(folder: str, filename: str) -> np.ndarray:
    return np.load(os.path.join(folder, filename), allow_pickle=True)

def get_ram_usage_in_gb(frames: int, column_size: int, nreps: int, row_size: int) -> int:
    '''
    Calculates the RAM usage in GB for a 4D array of the given dimensions.
    Args:
        frames: int
        column_size: int
        nreps: int
        row_size: int
    Returns:
        int
    '''
    return int(frames * column_size * nreps * row_size * 8 / 1024**3)