import numpy as np
from scipy.optimize import curve_fit
from numba import njit, prange

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


@njit(parallel=True)
def group_pixels(
    data, primary_threshold, secondary_threshold, noise_map, structure=None
):
    """
    Uses the two pass labelling to group events.
    Pixels over the primary threshold are connected to pixels above the
    secondary threshold according to a structure element.
    Input is of shape (frame,row,col), calulation over the frames is
    parallized using numba's prange.
    The output is a numpy array of shape (frame,row,col) with zeroes if there
    is no event above the primary threshold. Clustered events are labeled with
    integers beginning at 1.
    """
    output = np.zeros(data.shape, dtype=np.uint16)
    for frame_index in prange(data.shape[0]):
        mask_primary = data[frame_index] > primary_threshold * noise_map[frame_index]
        mask_secondary = (
            data[frame_index] > secondary_threshold * noise_map[frame_index]
        )
        # Set the first and last rows to zero
        mask_primary[0, :] = 0
        mask_primary[-1, :] = 0
        mask_secondary[0, :] = 0
        mask_secondary[-1, :] = 0

        # Set the first and last columns to zero
        mask_primary[:, 0] = 0
        mask_primary[:, -1] = 0
        mask_secondary[:, 0] = 0
        mask_secondary[:, -1] = 0

        if structure is None:
            structure = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        labeled_primary, num_features_primary = utils.two_pass_labeling(
            mask_primary, structure=structure
        )

        # Iterate over each feature in the primary mask
        for feature_num in range(1, num_features_primary + 1):
            # Create a mask for the current feature
            feature_mask = labeled_primary == feature_num

            # Expand the feature mask to include secondary threshold pixels
            expanded_mask = mask_secondary & feature_mask

            # Label the expanded mask
            labeled_expanded, _ = utils.two_pass_labeling(
                expanded_mask, structure=structure
            )

            # Get the indices where labeled_expanded > 0
            indices = np.where(labeled_expanded > 0)
            for i in range(len(indices[0])):
                output[frame_index, indices[0][i], indices[1][i]] = feature_num

    return output
