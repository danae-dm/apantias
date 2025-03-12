import numpy as np
from numba import njit, prange

from . import utils


def get_slopes(data: np.ndarray) -> np.ndarray:
    """
    Calculates the slope over nreps for every pixel and frame in parallel using numba.
    Args:
        data: np.array (nframes, column_size, nreps, row_size)
    Returns:
        slopes: np.array (nframes, column_size, row_size)
    """
    if np.ndim(data) != 4:
        raise ValueError("Input data is not a 4D array.")
    slopes = utils.apply_slope_fit_along_frames(data)
    return slopes


def correct_common_mode(data: np.ndarray) -> None:
    """
    Calculates the median of euch row in data, and substracts it from
    the row. The median is calculated in parallel using numba.
    Correction is done inline to save memory.
    Args:
        np.array in shape (nframes, column_size, nreps, row_size)
    """
    if data.ndim != 4:
        raise ValueError("Data is not a 4D array")
    # Iterate over the nframes dimension
    # Calculate the median for one frame
    # median_common = analysis_funcs.parallel_nanmedian_4d_axis3(data)
    # Subtract the median from the frame in-place
    median_common = utils.nanmedian(data, axis=3, keepdims=True)
    data -= median_common


@njit(parallel=True)
def group_pixels(data: np.ndarray, primary_threshold: float, secondary_threshold: float, noise_map: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """
    Uses the two pass labelling algorithm to group events.
    Pixels over the primary threshold are connected to pixels above the
    secondary threshold according to a structure element.
    Input is of shape (frame,row,col), calulation over the frames is
    parallized using numba's prange.
    The output is a numpy array of shape (frame,row,col) with zeroes if there
    is no event above the primary threshold. Clustered events are labeled with
    integers beginning with 1.
    """
    output = np.zeros(data.shape, dtype=np.uint16)
    for frame_index in prange(data.shape[0]):
        mask_primary = data[frame_index] > primary_threshold * noise_map
        mask_secondary = data[frame_index] > secondary_threshold * noise_map
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
