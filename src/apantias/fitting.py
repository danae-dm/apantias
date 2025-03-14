import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.optimize import curve_fit
import numpy as np
from numba import njit, prange

from . import utils


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
