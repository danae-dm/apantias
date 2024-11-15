import numpy as np
from scipy.optimize import curve_fit

from typing import List, Tuple
import gc

from . import logger
from . import fitting
from . import display
from . import utils

_logger = logger.Logger(__name__, "info").get_logger()


def exclude_mips_and_bad_frames(
    data: np.ndarray, thres_mips: int, thres_bad_frames: int
) -> np.ndarray:
    """
    Combines exclude_mips_frames and exclude_bad_frames to exclude frames
    that are above or below the median by a certain threshold.
    MIPS:
    Calculates the median of each frame and deletes frames where there is
    a pixel above or below the median by a certain threshold.
    BAD FRAMES:
    Calculates the average of each frame and excludes frames that are
    above or below the fitted mean by a certain threshold.
    Args:
        data: shape (nframes, column_size, nreps, row_size)
        thres_mips: absolute threshold in adu
        thres_bad_frames: used with the fitted sigma do exclude frames
    Returns:
        np.array in shape (nframes-X, column_size, nreps, row_size)
    """
    # TODO: rethink this. Its quite expensive to calculate the median/mean twice.
    # maybe just keep the more restrictive of the two?
    if np.ndim(data) != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    _logger.info(f"Excluding bad frames due to MIPS, threshold: {thres_mips}")
    _logger.info(f"Excluding bad frames, threshold: {thres_bad_frames}")
    median = utils.nanmedian(data, axis=3)
    median = utils.nanmedian(median, axis=2)
    median = utils.nanmedian(median, axis=1)
    # calculate mips mask
    mips_mask = (data > median[:, np.newaxis, np.newaxis, np.newaxis] + thres_mips) | (
        data < median[:, np.newaxis, np.newaxis, np.newaxis] - thres_mips
    )
    mips_mask = np.any(mips_mask, axis=(1, 2, 3))
    _logger.info(f"Excluded {np.sum(mips_mask)} frames due to mips")
    _logger.debug(f"Indices: {np.where(mips_mask)[0]}")
    # calculate bad frames mask
    mean = utils.nanmean(data, axis=3)
    mean = utils.nanmean(mean, axis=2)
    mean = utils.nanmean(mean, axis=1)
    fit = fitting.fit_gauss_to_hist(mean)
    lower_bound = fit[1] - thres_bad_frames * np.abs(fit[2])
    upper_bound = fit[1] + thres_bad_frames * np.abs(fit[2])
    bad_frames_mask = (mean < lower_bound) | (mean > upper_bound)
    _logger.info(f"Excluded {np.sum(bad_frames_mask)} bad frames")
    _logger.debug(f"Indices: {np.where(bad_frames_mask)[0]}")
    mask = mips_mask | bad_frames_mask
    return data[~mask]


def get_slopes(data: np.ndarray) -> np.ndarray:
    """
    Calculates the slope over nreps for every pixel and frame.
    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
    Returns:
        slopes: np.array (nframes, column_size, row_size) with the slope values
    """
    if np.ndim(data) != 4:
        _logger.error("Input data is not a 4D array.")
        raise ValueError("Input data is not a 4D array.")
    _logger.info("Calculating slope values")
    slopes = utils.apply_slope_fit_along_frames(data)
    _logger.info("Finished calculating slope values")
    _logger.debug(f"Shape of slopes: {slopes.shape}")
    return slopes


def set_bad_pixellist_to_nan(
    data: np.ndarray, bad_pixels: List[Tuple[int, int]]
) -> None:
    """
    Sets all ignored Pixels in data to NaN. List of pixels is from the
    parameter file. [(row,col), (row,col), ...]
    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
        bad_pixels: list of tuples (row,col)
    Returns:
        np.array in shape (nframes, column_size, nreps, row_size)
    """
    if np.ndim(data) != 4:
        _logger.error("Data is not a 4D array")
        raise ValueError("Data is not a 4D array")
    _logger.info("Excluding bad pixels")
    bad_pixel_mask = np.zeros(data.shape, dtype=bool)
    for index in bad_pixels:
        col = index[1]
        row = index[0]
        bad_pixel_mask[:, row, :, col] = True
    data[bad_pixel_mask] = np.nan
    _logger.info(f"Excluded {len(bad_pixels)} pixels")


def correct_common_mode(data: np.ndarray) -> None:
    """
    Calculates the median of euch row in data, and substracts it from
    the row.
    Correction is done inline to save memory.
    Args:
        np.array in shape (nframes, column_size, nreps, row_size)
    """
    if data.ndim != 4:
        _logger.error("Data is not a 4D array")
        raise ValueError("Data is not a 4D array")
    _logger.info("Starting common mode correction.")
    # Iterate over the nframes dimension
    # Calculate the median for one frame
    # median_common = analysis_funcs.parallel_nanmedian_4d_axis3(data)
    # Subtract the median from the frame in-place
    median_common = utils.nanmedian(data, axis=3, keepdims=True)
    data -= median_common
    _logger.info("Data is corrected for common mode.")


# TODO: Rewrite EventMap and GainFit:
# - EventMap: Current structure cannot be written to h5 file
# - GainFit: Find some parameters to find "good" pixels first


def calc_event_map(
    avg_over_nreps: np.ndarray, noise_fitted: np.ndarray, thres_event: int
) -> np.ndarray:
    """
    Finds events in the data by comparing the average over nreps with the
    noise fitted map.
    Args:
        avg_over_nreps: np.array (nframes, column_size, row_size)
        noise_fitted: np.array (column_size, row_size)
        thres_event: threshold for the events
    Returns:
        np.array (column_size, row_size) with lists of signals
    """
    if avg_over_nreps.ndim != 3:
        _logger.error("avg_over_nreps is not a 3D array")
        raise ValueError("avg_over_nreps is not a 3D array")
    if noise_fitted.ndim != 2:
        _logger.error("noise_fitted is not a 2D array")
        raise ValueError("noise_fitted is not a 2D array")
    _logger.info("Finding events")
    threshold_map = noise_fitted * thres_event
    events = avg_over_nreps > threshold_map[np.newaxis, :, :]
    signals = np.full(avg_over_nreps.shape, np.nan)
    signals[events] = avg_over_nreps[events]
    mask = ~np.isnan(signals)
    indices = np.argwhere(mask)
    values = signals[mask]
    _logger.info(f"{indices.shape[0]} events found")
    event_map = np.zeros((64, 64), dtype=object)
    for i in range(event_map.shape[0]):
        for j in range(event_map.shape[1]):
            event_map[i, j] = {"frame": [], "signal": []}
    for index, entry in enumerate(indices):
        frame = int(entry[0])
        row = int(entry[1])
        column = int(entry[2])
        signal = values[index]
        event_map[row][column]["frame"].append(frame)
        event_map[row][column]["signal"].append(signal)
    return event_map


def get_sum_of_event_signals(
    event_map: np.ndarray, row_size: int, column_size: int
) -> np.ndarray:
    """
    Adds the signal values of the events in the event_map
    Args:
        event_map: np.array (column_size, row_size)
        row_size: number of rows in the data
        column_size: number of columns in the data
    Returns:
        np.array (column_size, row_size) with the sum of the signals
    """
    sum_of_events = np.zeros((row_size, column_size))
    for row in range(row_size):
        for column in range(column_size):
            sum_of_events[row][column] = sum(event_map[row][column]["signal"])
    return sum_of_events


def get_sum_of_event_counts(
    event_map: np.ndarray, row_size: int, column_size: int
) -> np.ndarray:
    """
    Sums up the number of events in the event_map
    Args:
        event_map: np.array (column_size, row_size)
        row_size: number of rows in the data
        column_size: number of columns in the data
    Returns:
        np.array (column_size, row_size) with the count of the events
    """
    sum_of_events = np.zeros((row_size, column_size))
    for row in range(row_size):
        for column in range(column_size):
            sum_of_events[row][column] = len(event_map[row][column]["signal"])
    return sum_of_events


def get_gain_fit(
    event_map: np.ndarray, row_size: int, column_size: int, min_signals: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a gaussian to the histogram of the signals in the event_map. When
    there are fewer signals than min_signals, or the fit is not
    successfull, the values are set to np.nan.
    Args:
        event_map: np.array (column_size, row_size) with lists of signals
        row_size: number of rows in the data
        column_size: number of columns in the data
        min_signals: minimum number of signals for a fit
    Returns:
        np.array (column_size, row_size) with the mean of the gaussian
        np.array (column_size, row_size) with the sigma of the gaussian
        np.array (column_size, row_size) with the error of the mean
        np.array (column_size, row_size) with the error of the sigma
    """
    # TODO: optimize this
    mean = np.full((row_size, column_size), np.nan)
    sigma = np.full((row_size, column_size), np.nan)
    mean_error = np.full((row_size, column_size), np.nan)
    sigma_error = np.full((row_size, column_size), np.nan)

    def fit_hist(data):

        def gaussian(x, a1, mu1, sigma1):
            return a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))

        try:
            hist, bins = np.histogram(
                data, bins=10, range=(np.nanmin(data), np.nanmax(data)), density=True
            )
            bin_centers = (bins[:-1] + bins[1:]) / 2
            params, covar = curve_fit(
                gaussian, bin_centers, hist, p0=[1, bins[np.argmax(hist)], 1]
            )
            return (
                params[1],
                abs(params[2]),
                np.sqrt(np.diag(covar))[1],
                np.sqrt(np.diag(covar))[2],
            )
        except:
            return (np.nan, np.nan, np.nan, np.nan)

    count_too_few = 0
    for i in range(event_map.shape[0]):
        for j in range(event_map.shape[1]):
            signals = event_map[i, j]["signal"]
            if len(signals) >= min_signals:
                params = fit_hist(signals)
                mean[i, j] = params[0]
                sigma[i, j] = params[1]
                mean_error[i, j] = params[2]
                sigma_error[i, j] = params[3]
            else:
                count_too_few += 1
    _logger.info(f"{count_too_few} pixels have less than {min_signals} signals")
    return mean, sigma, mean_error, sigma_error
