import h5py
import numpy as np
import json
from typing import List, Tuple, Optional
import os

from . import logger

_logger = logger.Logger(__name__, "info").get_logger()


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
            count=column_size * raw_row_size * 2000 * 2 * 10,
            offset=offset,
        )
        test_data = test_data.reshape(-1, raw_row_size)
        frame_keys = np.where(test_data[:, column_size] == 65535)
        frames = np.stack((frame_keys[0][:-1], frame_keys[0][1:]))
        diff = np.diff(frames, axis=0)
        median_diff = np.median(diff)
        estimated_nreps = int(median_diff / column_size)
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
        _logger.warning("No valid frames found in chunk!")
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
    compression: str = None,
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
        compression
        available_ram_gb: the available RAM in GB
    """
    output_file = os.path.join(output_folder, output_filename)
    # check if h5 file already exists
    if os.path.exists(output_file):
        _logger.error(f"File {output_file} already exists. Please delete")
        raise Exception(f"File {output_file} already exists. Please delete")
    # check if bin files exist
    for bin_file in bin_files:
        if not os.path.exists(bin_file):
            _logger.error(f"File {bin_file} does not exist")
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
            compression=compression,
        )
        f.attrs["description"] = (
            "This file contains the raw data from the bin files, only complete frames are saved"
        )
        dataset.attrs["bin_files"] = bin_files
        dataset.attrs["column_size"] = column_size
        dataset.attrs["row_size"] = row_size
        dataset.attrs["nreps"] = nreps
        dataset.attrs["total_frames"] = 0
        if compression:
            dataset.attrs["compression"] = compression
        else:
            dataset.attrs["compression"] = "None"
        if attributes:
            for key, value in attributes.items():
                dataset.attrs[key] = value
        _logger.info(f"Initialized empty file: {output_file}")

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
            _logger.info(
                f"Loading file: {bin_file}\n size: {file_size_gb:.2f} GB\n estimated frames: {estimated_frames:.1f}\n chunk size: {chunk_size:.2f} GB"
            )
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
                    _logger.error(e)
                    _logger.error(f"Deleting file: {output_file}")
                    # delete the h5 file
                    os.remove(output_file)
                    break
                offset = new_offset
                if new_data is not None:
                    dataset.resize(dataset.shape[0] + new_data.shape[0], axis=0)
                    # Append the new data
                    dataset[-new_data.shape[0] :] = new_data
                    frames_loaded = dataset.shape[0]
                    _logger.info(
                        f"progress: {frames_loaded}/{estimated_frames:.0f} frames loaded ({frames_loaded/estimated_frames:.2%})"
                    )
        dataset.attrs["total_frames"] = dataset.shape[0]
        _logger.info(f"Finished loading data from bin files to {output_file}")


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


def get_params_from_data_file(file_path: str) -> Tuple[int, int, int, int]:
    """
    Get the parameters from the data h5 file.
    """
    with h5py.File(file_path, "r") as file:
        total_frames = file["data"].shape[0]
        column_size = file["data"].attrs["column_size"]
        row_size = file["data"].attrs["row_size"]
        nreps = file["data"].attrs["nreps"]
    return total_frames, column_size, row_size, nreps


def create_analysis_file(
    output_folder: str,
    output_filename: str,
    offnoi_data_file: str,
    filter_data_file: str,
    parameter_file_contents: dict,
) -> None:
    """
    Create an analysis h5 file with offnoi/filter/gain groups.
    An existing data file must be provided for offnoi and filter.
    The parameter file contents are saved as a json string.
    """

    output_file = os.path.join(output_folder, output_filename)
    if os.path.exists(output_file):
        raise Exception(f"File {output_file} already exists. Please delete")
    # raise excption when data files dont exist
    if os.path.exists(offnoi_data_file) == False:
        _logger.error(f"File {offnoi_data_file} does not exist.")
        raise Exception(f"File {offnoi_data_file} does not exist.")
    if os.path.exists(filter_data_file) == False:
        _logger.error(f"File {filter_data_file} does not exist.")
        raise Exception(f"File {filter_data_file} does not exist.")
    # create the hdf5 file
    with h5py.File(output_file, "w") as f:
        f.attrs["description"] = "This file contains the results of the analysis."

        f.create_group("offnoi")
        f["offnoi"].attrs["data"] = offnoi_data_file
        f["offnoi"].attrs[
            "description"
        ] = "This group contains the results of the offset noise analysis."
        with h5py.File(offnoi_data_file, "r") as offnoi_data_file:
            f["offnoi"].attrs["bin_files"] = offnoi_data_file["data"].attrs["bin_files"]
            f["offnoi"].attrs["column_size"] = offnoi_data_file["data"].attrs[
                "column_size"
            ]
            f["offnoi"].attrs["row_size"] = offnoi_data_file["data"].attrs["row_size"]
            f["offnoi"].attrs["nreps"] = offnoi_data_file["data"].attrs["nreps"]
            f["offnoi"].attrs["total_frames"] = offnoi_data_file["data"].shape[0]

        f.create_group("filter")
        f["filter"].attrs["data"] = filter_data_file
        f["filter"].attrs[
            "description"
        ] = "This group contains the results of the filter analysis."
        with h5py.File(filter_data_file, "r") as filter_data_file:
            f["filter"].attrs["bin_files"] = filter_data_file["data"].attrs["bin_files"]
            f["filter"].attrs["column_size"] = filter_data_file["data"].attrs[
                "column_size"
            ]
            f["filter"].attrs["row_size"] = filter_data_file["data"].attrs["row_size"]
            f["filter"].attrs["nreps"] = filter_data_file["data"].attrs["nreps"]
            f["filter"].attrs["total_frames"] = filter_data_file["data"].shape[0]

        f.create_group("gain")
        f["gain"].attrs[
            "description"
        ] = "This group contains the results of the gain analysis."
        f.create_group("parameter_json")

        f["parameter_json"].attrs[
            "description"
        ] = "This group contains the parameter file used for the analysis."
        f.create_dataset(
            "parameter_json/parameter_file",
            data=repr(parameter_file_contents),
            dtype=h5py.special_dtype(vlen=str),
        )
        _logger.info(f"Initialized empty file: {output_file}")


def get_data_from_file(
    file_path: str,
    dataset_path: str,
    slices: List[slice] = None,
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
    with h5py.File(file_path, "r") as file:
        dataset = file[dataset_path]
        if slices is not None:
            if dataset.ndim != len(slices):
                _logger.error(
                    f"Dataset has {dataset.ndim} dimensions, but {len(slices)} slices were provided."
                )
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
    file_path: str,
    dataset_path: str,
    data: np.ndarray,
    attributes: dict = None,
    compression: str = None,
) -> None:
    """
    Adds an array to a Dataset. If the dataset exists, the array is appended.
    Dataset and group is created if not already present.
    Per default no compression is used, bit 'gzip' can be specified.

    Args:
        file_path: Path to the HDF5 file.
        group_name: Name of the group.
        dataset_name: Name of the dataset.
        data: Data to save.
        attributes: Attributes to save.
        compression: Compression to use.
    """
    with h5py.File(file_path, "a") as file:
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

        current_group.create_dataset(
            dataset_name,
            data=data,
            maxshape=(None, *data.shape[1:]),
            chunks=(1, *data.shape[1:]),
            compression=compression,
        )
        current_dataset = current_group[dataset_name]
        if attributes:
            for key, value in attributes.items():
                current_dataset.attrs[key] = value

        if current_dataset.shape[1:] != data.shape[1:]:
            _logger.error(
                f"Shape of data to add ({data.shape[1:]}) does not match shape of existing dataset ({current_dataset.shape[1:]})"
            )
            raise Exception(
                f"Shape of data to add ({data.shape[1:]}) does not match shape of existing dataset ({current_dataset.shape[1:]})"
            )
        if dataset_name == "offset_raw":
            # the data here should not be stacked but averaged
            current_dataset = (current_dataset + data) / 2
            if attributes:
                for key, value in attributes.items():
                    current_dataset.attrs[key] = value
        else:
            current_dataset.resize(current_dataset.shape[0] + data.shape[0], axis=0)
            current_dataset[-data.shape[0] :] = data
            if attributes:
                for key, value in attributes.items():
                    current_dataset.attrs[key] = value
