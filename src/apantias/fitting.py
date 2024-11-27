import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.optimize import curve_fit
import numpy as np
from numba import njit, prange

from . import logger

_logger = logger.Logger(__name__, "info").get_logger()


# TODO: Optimize this by not computing the guesses (calculate them beforehand und save them to a file)
def fit_gauss_to_hist(data_to_fit: np.ndarray) -> np.ndarray:
    """
    fits a gaussian to a histogram using the scipy curve_fit method

    Args:
        data_to_fit: np.array in 1 dimension
    Returns:
        np.array[amplitude, mean, sigma, error_amplitude, error_mean, error_sigma]
    """
    if np.all(np.isnan(data_to_fit)):
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    try:
        min = np.nanmin(data_to_fit)
        max = np.nanmax(data_to_fit)
        hist, bins = np.histogram(
            data_to_fit,
            bins=100,
            range=(min, max),
            density=True,
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        median = np.nanmedian(data_to_fit)
        std = np.nanstd(data_to_fit)
        ampl_guess = np.nanmax(hist)
        guess = [ampl_guess, median, std]
        bounds = (
            [0, min, 0],
            [np.inf, max, np.inf],
        )
        params, covar = curve_fit(gaussian, bin_centers, hist, p0=guess, bounds=bounds)
        return np.array(
            [
                params[0],
                params[1],
                np.abs(params[2]),
                np.sqrt(np.diag(covar))[0],
                np.sqrt(np.diag(covar))[1],
                np.sqrt(np.diag(covar))[2],
            ]
        )
    except:
        _logger.debug("Fitting for this histogram failed. Returning NaNs.")
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def fit_2_gauss_to_hist(data_to_fit: np.ndarray) -> np.ndarray:
    """
    fits a double gaussian to a histogram using the scipy curve_fit method

    Args:
        data_to_fit: np.array in 1 dimension
    Returns:
        np.array[amplitude1, mean1, sigma1, error_amplitude1, error_mean1, error_sigma1,
        amplitude2, mean2, sigma2, error_amplitude2, error_mean2, error_sigma2]
    """
    if np.all(np.isnan(data_to_fit)):
        return np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )

    try:
        min = np.nanmin(data_to_fit)
        max = np.nanmax(data_to_fit)
        hist, bins = np.histogram(
            data_to_fit,
            bins=100,
            range=(min, max),
            density=True,
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        median = np.nanmedian(data_to_fit)
        std = np.nanstd(data_to_fit)
        ampl_guess = np.nanmax(hist)
        guess = [ampl_guess, median, std, 0.3 * ampl_guess, median + 1, std]
        bounds = (
            [0, min, 0, 0, min, 0],
            [np.inf, max, np.inf, np.inf, max, np.inf],
        )
        params, covar = curve_fit(
            two_gaussians, bin_centers, hist, p0=guess, bounds=bounds
        )
        return np.array(
            [
                params[0],
                params[1],
                np.abs(params[2]),
                np.sqrt(np.diag(covar))[0],
                np.sqrt(np.diag(covar))[1],
                np.sqrt(np.diag(covar))[2],
                params[3],
                params[4],
                np.abs(params[5]),
                np.sqrt(np.diag(covar))[3],
                np.sqrt(np.diag(covar))[4],
                np.sqrt(np.diag(covar))[5],
            ]
        )
    except:
        _logger.debug("Fitting for this histogram failed. Returning NaNs.")
        return np.array(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        )


def process_row(data, row, peaks):
    if peaks not in [1, 2]:
        raise ValueError("Peaks must be 1 or 2")
    if peaks == 1:
        result = np.apply_along_axis(fit_gauss_to_hist, axis=0, arr=data)
    if peaks == 2:
        result = np.apply_along_axis(fit_2_gauss_to_hist, axis=0, arr=data)
    return row, result


def get_pixelwise_fit(data: np.ndarray, peaks: int) -> np.ndarray:
    # apply the function to every pixel
    if peaks not in [1, 2]:
        raise ValueError("Peaks must be 1 or 2")
    if peaks == 1:
        results = np.zeros((data.shape[1], data.shape[2], 6))
        with ProcessPoolExecutor() as executor:
            # add futures
            futures = []
            for row in range(data.shape[1]):
                row_data = np.copy(data[:, row, :])
                futures.append(executor.submit(process_row, row_data, row, 1))
                _logger.debug(f"Added row {row} to executor")
            total_futures = len(futures)
            completed_futures = 0
            for future in as_completed(futures):
                try:
                    row, row_results = future.result()
                    results[row] = row_results.T
                    completed_futures += 1
                    _logger.debug(
                        f"Completed row {row} ({completed_futures}/{total_futures})"
                    )
                except Exception as e:
                    _logger.error(f"Error processing column {row}: {e}")
        return results
    if peaks == 2:
        results = np.zeros((data.shape[1], data.shape[2], 12))
        with ProcessPoolExecutor() as executor:
            # add futures
            futures = []
            for row in range(data.shape[1]):
                row_data = np.copy(data[:, row, :])
                futures.append(executor.submit(process_row, row_data, row, 2))
                _logger.debug(f"Added row {row} to executor")
            total_futures = len(futures)
            completed_futures = 0
            for future in as_completed(futures):
                try:
                    row, row_results = future.result()
                    results[row] = row_results.T
                    completed_futures += 1
                    _logger.debug(
                        f"Completed row {row} ({completed_futures}/{total_futures})"
                    )
                except Exception as e:
                    _logger.error(f"Error processing column {row}: {e}")
        return results


def get_fit_over_frames(data: np.ndarray, peaks: int) -> np.ndarray:
    """
    fits a gaussian for every pixel. The fitting is done over the
    histogram of the pixel values from all frames using the scipy
    curve_fit method.

    Args:
        data: in shape (nframes, column_size, row_size)
    Returns:
        np.array in shape (6, rows, columns)
        index 0: amplitude
        index 1: mean
        index 2: sigma
        index 3: error_amplitude
        index 4: error_mean
        index 5: error_sigma
    """
    if data.ndim != 3:
        _logger.error("Data is not a 3D array")
        raise ValueError("Data is not a 3D array")
    if peaks not in [1, 2]:
        _logger.error("Peaks must be 1 or 2")
        raise ValueError("Peaks must be 1 or 2")

    # apply the function to every frame
    if peaks == 1:
        output = np.apply_along_axis(fit_gauss_to_hist, axis=0, arr=data)
    if peaks == 2:
        output = np.apply_along_axis(fit_2_gauss_to_hist, axis=0, arr=data)
    return output


def gaussian(x: float, a: float, mu: float, sigma: float) -> float:
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def two_gaussians(
    x: float, a1: float, mu1: float, sigma1: float, a2: float, mu2: float, sigma2: float
) -> float:
    return gaussian(x, a1, mu1, sigma1) + gaussian(x, a2, mu2, sigma2)


@njit(parallel=False)
def linear_fit(data: np.ndarray) -> np.ndarray:
    """
    Fits a linear function to the data using the least squares method.
    """
    x = np.arange(data.size)
    n = data.size

    # Calculate the sums needed for the linear fit
    sum_x = np.sum(x)
    sum_y = np.sum(data)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * data)

    # Calculate the slope (k) and intercept (d)
    k = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    d = (sum_y - k * sum_x) / n

    return np.array([k, d])
