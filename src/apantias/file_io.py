import h5py
import numpy as np
import json
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import gc

from . import utils
from . import __version__


def read_data_chunk_from_bin(
    bin_file: str,
    column_size: int,
    row_size: int,
    key_ints: int,
    nreps: int,
    frames_to_read: int,
    offset: int,
) -> Tuple[np.ndarray, int]:
    """
    Reads data from a .bin file in chunks and returns a numpy array with the data and the new offset.
    The bins usually have a header of 8 bytes, followed by the data. So an initial offset of 8 is needed.
    After that the new offset is the position in the file where the next chunk should start. The parameter
    frames_to_read should be set to a value that fits into the available RAM.
    The function returns -1 as offset when end of file is reached.

    Returns:
        data, offset: Tuple[np.ndarray, int]
    """
    raw_row_size = row_size + key_ints
    # test if nreps make sense
    if offset == 8:
        test_data = np.fromfile(
            bin_file,
            dtype="uint16",
            # load three times the given nreps and load 20 frames, times two because uint16 is 2 bytes
            count=column_size * raw_row_size * 3 * nreps * 10 * 2,
            offset=offset,
        )
        test_data = test_data.reshape(-1, raw_row_size)
        # get indices of frame keys, they are in the last column
        frame_keys = np.where(test_data[:, column_size] == 65535)
        frames = np.stack((frame_keys[0][:-1], frame_keys[0][1:]))
        # calculate distances between frame keys
        diff = np.diff(frames, axis=0)
        # determine which distance is the most common
        unique_numbers, counts = np.unique(diff, return_counts=True)
        max_count_index = np.argmax(counts)
        estimated_distance = unique_numbers[max_count_index]
        estimated_nreps = int(estimated_distance / column_size)
        if nreps != estimated_nreps:
            raise Exception(f"Estimated nreps: {estimated_nreps}, given nreps: {nreps}")

    raw_frame_size = column_size * raw_row_size * nreps
    rows_per_frame = column_size * nreps
    chunk_size = raw_frame_size * frames_to_read

    inp_data = np.fromfile(bin_file, dtype="uint16", count=chunk_size, offset=offset)
    offset += chunk_size * 2  # offset is in bytes, uint16 = 16 bit = 2 bytes
    # check if file is at its end
    if inp_data.size == 0:
        return None, -1
    # reshape the array into rows -> (#ofRows,67)
    inp_data = inp_data.reshape(-1, raw_row_size)
    # find all the framekeys
    frame_keys = np.where(inp_data[:, column_size] == 65535)
    # stack them and calculate difference to find incomplete frames
    frames = np.stack((frame_keys[0][:-1], frame_keys[0][1:]))
    diff = np.diff(frames, axis=0)
    valid_frames_position = np.nonzero(diff == rows_per_frame)[1]
    if len(valid_frames_position) == 0:
        return None, offset
    valid_frames = frames.T[valid_frames_position]
    frame_start_indices = valid_frames[:, 0]
    frame_end_indices = valid_frames[:, 1]
    inp_data = np.array(
        [
            inp_data[start + 1 : end + 1, :64]
            for start, end in zip(frame_start_indices, frame_end_indices)
        ]
    )
    inp_data = inp_data.reshape(-1, column_size, nreps, row_size)
    return inp_data, offset


def create_data_file_from_bins(
    bin_files: List[str],
    output_folder: str,
    output_filename: str,
    nreps: int,
    column_size: int = 64,
    row_size: int = 64,
    attributes: dict = None,
    available_ram_gb: int = 16,
) -> None:
    """
    This function creates a data file in h5 format from a list of bin files.
    It reads chunks from the bin files to memory and writes them to the h5 file.
    The chunk size is determined by the available RAM in GB. A chunk of 30% of
    the available RAM is read at once.
    Args:
        bin_files: absolute paths to the bin files
        output_folder: absolute path to a output folder
        output_filename
        nreps
        column_size
        row_size
        attributes
        available_ram_gb: the available RAM in GB
    """
    output_file = os.path.join(output_folder, output_filename)
    # check if h5 file already exists
    if os.path.exists(output_file):
        raise Exception(f"File {output_file} already exists. Please delete")
    # check if bin files exist
    for bin_file in bin_files:
        if not os.path.exists(bin_file):
            raise Exception(f"File {bin_file} does not exist")
    # create the hdf5 file
    with h5py.File(output_file, "w") as f:
        # create the dataset
        dataset = f.create_dataset(
            "data",
            dtype="uint16",
            shape=(0, column_size, nreps, row_size),
            maxshape=(None, column_size, nreps, row_size),
            chunks=(1, column_size, nreps, row_size),
        )
        f.attrs["description"] = (
            "This file contains the raw data from the bin files, only complete frames are saved"
        )
        dataset.attrs["bin_files"] = bin_files
        dataset.attrs["column_size"] = column_size
        dataset.attrs["row_size"] = row_size
        dataset.attrs["nreps"] = nreps
        dataset.attrs["total_frames"] = 0
        if attributes:
            for key, value in attributes.items():
                dataset.attrs[key] = value

        for bin_file in bin_files:
            file_size = os.path.getsize(bin_file)
            frame_size_bytes = column_size * row_size * nreps * 2
            file_size_gb = file_size / (1024 * 1024 * 1024)
            estimated_frames = file_size / frame_size_bytes
            # determine how many frames to read at once
            frames_to_read = int(
                (available_ram_gb * 1024 * 1024 * 1024 / frame_size_bytes) * 0.3
            )
            chunk_size = (frames_to_read * frame_size_bytes) / (1024 * 1024 * 1024)
            offset = 8
            while offset != -1:  # get_data returns -1 as offset when EOF is reached
                try:
                    new_data, new_offset = read_data_chunk_from_bin(
                        bin_file,
                        column_size,
                        row_size,
                        3,
                        nreps,
                        frames_to_read,
                        offset,
                    )
                except Exception as e:
                    raise Exception(f"Error reading data from bin file: {e}")
                finally:
                    os.remove(output_file)
                offset = new_offset
                if new_data is not None:
                    dataset.resize(dataset.shape[0] + new_data.shape[0], axis=0)
                    # Append the new data
                    dataset[-new_data.shape[0] :] = new_data
                    frames_loaded = dataset.shape[0]
        dataset.attrs["total_frames"] = dataset.shape[0]


def display_file_structure(file_path: str) -> None:
    """
    Displays the structure (groups and datasets) of an HDF5 file.
    """

    def print_structure(name, obj):
        indent = "  " * (name.count("/") - 1)
        if isinstance(obj, h5py.Group):
            print(f"{indent}Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")

        # Print attributes
        for key, value in obj.attrs.items():
            print(f"{indent}  Attribute: {key} = {value}")

    with h5py.File(file_path, "r") as file:
        file.visititems(print_structure)


def get_data_from_file(
    path: str,
    dataset_path: str = None,
    slice: str = None,
) -> np.ndarray:
    """
    Get the data from the HDF5 file.

    Args:
        file_path: Path to the HDF5 file.
        dataset_path: Path to the data set in the HDF5 file.
        slices: List of slices to apply to the dataset. If None, the whole dataset is returned.
    Returns:
        data: np.ndarray
    """
    if dataset_path is None:
        file_path, dataset_path = utils.split_h5_path(path)
    else:
        file_path = path
        dataset_path = dataset_path
    if slice is not None:
        slices = utils.parse_numpy_slicing(slice)
    else:
        slices = None

    with h5py.File(file_path, "r") as file:
        dataset = file[dataset_path]
        if slices is not None:
            if dataset.ndim != len(slices):
                raise Exception(
                    f"Dataset has {dataset.ndim} dimensions, but {len(slices)} slices were provided."
                )
            else:
                data = dataset[tuple(slices)]
        else:
            data = dataset[:]

        data = data.astype(np.float64)
    return data


def add_array(
    path: str, data: np.ndarray, dataset_path: str = None, attributes: dict = None
) -> None:
    """
    Adds an array to a Dataset. If the dataset exists, the array is appended.
    Dataset and group is created if not already present.

    Args:
        file_path: Path to the HDF5 file.
        group_name: Name of the group.
        dataset_name: Name of the dataset.
        data: Data to save.
        attributes: Attributes to save.
    """
    if dataset_path is None:
        file_path, dataset_path = utils.split_h5_path(file_path)
    else:
        file_path = path
        dataset_path = dataset_path
    with h5py.File(file_path, "a", libver="latest") as file:
        # Split the dataset path into groups and dataset name
        parts = dataset_path.split("/")
        groups = parts[:-1]
        dataset_name = parts[-1]
        # Create groups if they do not exist
        current_group = file
        for group in groups:
            if group not in current_group:
                current_group = current_group.create_group(group)
            else:
                current_group = current_group[group]
        # Check if the dataset already exists
        if dataset_name not in current_group:
            # Create the new dataset in the group
            current_dataset = current_group.create_dataset(
                dataset_name,
                shape=(0, *data.shape[1:]),
                maxshape=(None, *data.shape[1:]),
                dtype=data.dtype,
            )
        else:
            current_dataset = current_group[dataset_name]

        # append data to existing dataset
        if current_dataset.shape[1:] != data.shape[1:]:
            raise Exception(
                f"Shape of data to add ({data.shape[1:]}) does not match shape of existing dataset ({current_dataset.shape[1:]})"
            )
        current_dataset.resize(current_dataset.shape[0] + data.shape[0], axis=0)
        current_dataset[-data.shape[0] :] = data
        if attributes:
            for key, value in attributes.items():
                current_dataset.attrs[key] = value


def _get_params_from_data_file(file_path: str) -> Tuple[int, int, int, int]:
    """
    Get the parameters from the data h5 file.
    """
    with h5py.File(file_path, "r") as file:
        total_frames = file["preproc_common_modes"].shape[0]
        column_size = file["preproc_mean_nreps"].shape[2]
        row_size = file["preproc_common_modes"].shape[1]
        nreps = file["preproc_common_modes"].shape[2]
    return total_frames, column_size, row_size, nreps


def _create_analysis_file(
    output_folder: str,
    output_filename: str,
    parameter_file_contents: dict,
    attributes_dict: dict,
) -> None:
    """
    Create an analysis h5 file with offnoi/filter/gain groups.
    An existing data file must be provided for offnoi and filter.
    The parameter file contents are saved as a json string.
    """

    output_file = os.path.join(output_folder, output_filename)
    if os.path.exists(output_file):
        raise Exception(f"File {output_file} already exists. Please delete")
    # create the hdf5 file
    with h5py.File(output_file, "w") as f:
        if attributes_dict:  # an empty dict evaluates to False
            f.attrs["description"] = (
                "This file contains the results of the analysis.\n No additional information has been provided in the parameter file."
            )
        else:
            for key, value in attributes_dict.items():
                f.attrs[key] = value
        f.create_group("0_infos")
        f.create_dataset(
            "0_infos/parameter_json",
            data=repr(parameter_file_contents),
            dtype=h5py.special_dtype(vlen=str),
        )
        f.create_dataset(
            "0_infos/apantias_version",
            data=repr(__version__),
            dtype=h5py.special_dtype(vlen=str),
        )
