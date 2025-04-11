"""
This module provides functions for fitting data to mathematical models, including single and double Gaussian fits
and linear fits. It leverages SciPy for curve fitting and Numba for optimized numerical computations. The module
is designed to handle noisy data and return both fitted parameters and their associated errors.
"""

from scipy.optimize import curve_fit
import numpy as np
from numba import njit


def fit_gauss_to_hist(data_to_fit: np.ndarray) -> np.ndarray:
    """
    Fits a single Gaussian to a histogram using the SciPy curve_fit method.

    Args:
        data_to_fit (np.ndarray): 1D array of data to fit.

    Returns:
        np.ndarray: Array containing the fitted parameters and their errors:
                    [amplitude, mean, sigma, error_amplitude, error_mean, error_sigma].
    """
    if np.all(np.isnan(data_to_fit)):
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    try:
        data_min = np.nanmin(data_to_fit)
        data_max = np.nanmax(data_to_fit)
        hist, bins = np.histogram(
            data_to_fit,
            bins=100,
            range=(data_min, data_max),
            density=True,
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        median = np.nanmedian(data_to_fit)
        std = np.nanstd(data_to_fit)
        ampl_guess = np.nanmax(hist)
        guess = [ampl_guess, median, std]
        bounds = (
            [0, data_min, 0],
            [np.inf, data_max, np.inf],
        )
        result = curve_fit(gaussian, bin_centers, hist, p0=guess, bounds=bounds)
        params, covar = result[:2]
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
    except (ValueError, RuntimeError):
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def fit_2_gauss_to_hist(data_to_fit: np.ndarray) -> np.ndarray:
    """
    Fits a double Gaussian to a histogram using the SciPy curve_fit method.

    Args:
        data_to_fit (np.ndarray): 1D array of data to fit.

    Returns:
        np.ndarray: Array containing the fitted parameters and their errors:
                    [amplitude1, mean1, sigma1, error_amplitude1, error_mean1, error_sigma1,
                     amplitude2, mean2, sigma2, error_amplitude2, error_mean2, error_sigma2].
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
        data_min = np.nanmin(data_to_fit)
        data_max = np.nanmax(data_to_fit)
        hist, bins = np.histogram(
            data_to_fit,
            bins=100,
            range=(data_min, data_max),
            density=True,
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        median = np.nanmedian(data_to_fit)
        std = np.nanstd(data_to_fit)
        ampl_guess = np.nanmax(hist)
        guess = [ampl_guess, median, std, 0.3 * ampl_guess, median + 1, std]
        bounds = (
            [0, data_min, 0, 0, data_min, 0],
            [np.inf, data_max, np.inf, np.inf, data_max, np.inf],
        )
        result = curve_fit(two_gaussians, bin_centers, hist, p0=guess, bounds=bounds)
        params, covar = result[:2]
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
    except (ValueError, RuntimeError):
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


def gaussian(x: float | np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """
    Computes the value of a Gaussian function.

    Args:
        x (float): Input value.
        a (float): Amplitude of the Gaussian.
        mu (float): Mean of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        float: Value of the Gaussian function at x.
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def two_gaussians(
    x: float | np.ndarray,
    a1: float,
    mu1: float,
    sigma1: float,
    a2: float,
    mu2: float,
    sigma2: float,
) -> np.ndarray:
    """
    Computes the sum of two Gaussian functions.

    Args:
        x (float): Input value.
        a1 (float): Amplitude of the first Gaussian.
        mu1 (float): Mean of the first Gaussian.
        sigma1 (float): Standard deviation of the first Gaussian.
        a2 (float): Amplitude of the second Gaussian.
        mu2 (float): Mean of the second Gaussian.
        sigma2 (float): Standard deviation of the second Gaussian.

    Returns:
        float: Value of the sum of the two Gaussian functions at x.
    """
    return gaussian(x, a1, mu1, sigma1) + gaussian(x, a2, mu2, sigma2)


@njit(parallel=False)
def linear_fit(data: np.ndarray) -> np.float64:
    """
    Fits a linear function to the data using the least squares method.

    Args:
        data (np.ndarray): 1D array of data to fit.

    Returns:
        np.float64: Slope of the fitted linear function.
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
    # d = (sum_y - k * sum_x) / n
    # return np.array([k, d])
    return k
