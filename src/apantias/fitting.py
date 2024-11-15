import gc

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
    guess = [1, np.nanmedian(data_to_fit), np.nanstd(data_to_fit)]
    try:
        hist, bins = np.histogram(
            data_to_fit,
            bins=100,
            range=(np.nanmin(data_to_fit), np.nanmax(data_to_fit)),
            density=True,
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        params, covar = curve_fit(gaussian, bin_centers, hist, p0=guess)
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


def get_fit_gauss(data: np.ndarray) -> np.ndarray:
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
    # apply the function to every frame
    output = np.apply_along_axis(fit_gauss_to_hist, axis=0, arr=data)
    return output


def gaussian(x: float, a1: float, mu1: float, sigma1: float) -> float:
    return a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))


def two_gaussians(
    x: float, a1: float, mu1: float, sigma1: float, a2: float, mu2: float, sigma2: float
) -> float:
    return a1 * np.exp(
        -((x - mu1) ** 2) / (2 * sigma1**2)
        + a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
    )


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
