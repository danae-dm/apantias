import gc

from scipy.optimize import curve_fit
import numpy as np
from iminuit import cost, Minuit

from . import logger

_logger = logger.Logger(__name__, 'info').get_logger()

def fit_gauss_to_hist(data_to_fit: np.ndarray) -> np.ndarray:
    ''' 
    fits a gaussian to a histogram of data_to_fit
    Args:
        1D np.array
    Returns:
        np.array[amplitude, mean, sigma, error_amplitude, error_mean, error_sigma]
    '''
    guess = [1, np.nanmedian(data_to_fit), np.nanstd(data_to_fit)]
    try:
        hist, bins = np.histogram(data_to_fit, bins=100, range=(np.nanmin(data_to_fit), np.nanmax(data_to_fit)), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        params, covar = curve_fit(gaussian, bin_centers, hist, p0=guess)
        return np.array([params[0],
                         params[1], 
                         np.abs(params[2]),
                         np.sqrt(np.diag(covar))[0],
                         np.sqrt(np.diag(covar))[1], 
                         np.sqrt(np.diag(covar))[2]])
    except:
        _logger.debug('Fitting for this histogram failed. Returning NaNs.')
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    
def unbinned_fit_gauss_to_hist(data_to_fit: np.ndarray) -> np.ndarray:
    '''
    fits a gaussian to a histogram of data_to_fit
    Args:
        1D np.array
    Returns:
        np.array[amplitude, mean, sigma, error_mean, error_sigma]
    '''
    #TODO: this doesnt seem to work: test this!
    c = cost.UnbinnedNLL(data_to_fit, gaussian)
    m = Minuit(c, 
               a1=1, 
               mu1=np.nanmedian(data_to_fit), 
               sigma1=np.nanstd(data_to_fit))
    m.limits['a1'] = (0, 100)
    m.limits['mu1'] = (np.nanmin(data_to_fit), np.nanmax(data_to_fit))
    m.limits['sigma1'] = (0, 100)
    m.migrad()
    if not m.valid:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    return np.array([m.values['a1'],
                     m.values['mu1'],
                     np.abs(m.values['sigma1']),
                     m.errors['a1'],
                     m.errors['mu1'],
                     m.errors['sigma1']])

def get_fit_gauss(data: np.ndarray) -> np.ndarray:
    ''' 
    fits a gaussian to a histogram of data_to_fit
    using the scipy curve_fit method
    Args:
        np.array in shape (nframes, column_size, row_size)
    Returns:
        np.array in shape (6, rows, columns)
        index 0: amplitude
        index 1: mean
        index 2: sigma
        index 3: error_amplitude
        index 4: error_mean
        index 5: error_sigma
    '''
    if data.ndim != 3:
        _logger.error('Data is not a 3D array')
        raise ValueError('Data is not a 3D array')
    #apply the function to every frame
    output = np.apply_along_axis(fit_gauss_to_hist, axis = 0, arr = data)
    return output

def get_unbinned_fit_gauss(data: np.ndarray) -> np.ndarray:
    '''
    fits a gaussian to a histogram of data
    using the unbinned method in minuit
    returns a np.array in shape (6, rows, columns)
    Args:
        np.array in shape (nframes, column_size, row_size)
    Returns:
        np.array in shape (6, rows, columns)
        index 0: amplitude
        index 1: mean
        index 2: sigma
        index 3: error_amplitude
        index 4: error_mean
        index 5: error_sigma
    '''
    if data.ndim != 3:
        _logger.error('Data is not a 3D array')
        raise ValueError('Data is not a 3D array')
    #apply the function to every frame
    output = np.apply_along_axis(unbinned_fit_gauss_to_hist, axis = 0, arr = data)
    return output

def gaussian(x: float, a1: float, mu1: float, sigma1: float) -> float:
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)))

def two_gaussians(x: float, a1: float, mu1: float, sigma1: float, 
                  a2: float, mu2: float, sigma2: float) -> float:
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2) +
            a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))))

def linear_fit(data: np.ndarray) -> np.ndarray:
    x = np.arange(data.size)
    k, d = np.polyfit(x, data, 1)
    return np.array([k, d])