import h5py
import numpy as np
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

    Args:
        bin_file: str
        column_size: int
        row_size: int
        key_ints: int
        nreps: int
        frames_to_read: int
        offset: int
    Returns:
        data, offset: Tuple[np.ndarray, int]
    """
    raw_row_size = row_size + key_ints
    raw_frame_size = column_size * raw_row_size * nreps
    rows_per_frame = column_size * nreps
    chunk_size = raw_frame_size * frames_to_read

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
        raise Exception("No valid frames found in chunk, wrong nreps?")
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
    Read data from bin files
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


def display_file_structure(file_path: str) -> None:
    """
    Displays the structure of an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
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


def get_params_from_data_file(file_path: str) -> Tuple[int, int, int]:
    """
    Get the parameters from the data h5 file.

    Args:
        file_path (str): Path to the HDF5 file.
    Returns:
        column_size, row_size, nreps: Tuple[int, int, int]
    """
    with h5py.File(file_path, "r") as file:
        total_frames = file["data"].attrs["total_frames"]
        column_size = file["data"].attrs["column_size"]
        row_size = file["data"].attrs["row_size"]
        nreps = file["data"].attrs["nreps"]
    return total_frames, column_size, row_size, nreps


def create_analysis_file(
    output_folder, output_filename, offnoi_data_file, filter_data_file
) -> None:
    """
    Create an analysis h5 file with the correct structure.
    This must be provided with an existing data file.
    """
    output_file = os.path.join(output_folder, output_filename)
    if os.path.exists(output_file):
        raise Exception(f"File {output_file} already exists. Please delete")
    # raise excption when data files dont exist
    if os.path.exists(offnoi_data_file) == False:
        raise Exception(f"File {offnoi_data_file} does not exist.")
    if os.path.exists(filter_data_file) == False:
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
        _logger.info(f"Initialized empty file: {output_file}")


def get_data_from_file(
    file_path: str,
    group_name: str,
    dataset_name: str,
    slicing: str = None,
) -> np.ndarray:
    """
    Get the data from the HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset.
        slicing (str): Numpy slicing string (e.g., "[0:10, :, 50:100, :]").
    Returns:
        data: np.ndarray
    """
    with h5py.File(file_path, "r") as file:
        dataset = file[f"{group_name}/{dataset_name}"]
        if slicing is not None:
            print(_parse_numpy_slicing(slicing))
            data = dataset[_parse_numpy_slicing(slicing)]
        else:
            data = dataset[:]

        data = data.astype(np.float64)
    return data


def create_dataset_in_group(
    file_path: str,
    group_name: str,
    dataset_name: str,
    data: np.ndarray,
    shape: Tuple[int] = None,
    attributes: dict = None,
    compression: str = None,
) -> None:
    """
    Create a dataset in a specific group within an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        group_name (str): Name of the group.
        dataset_name (str): Name of the dataset.
        data (np.ndarray): Data to save.
        shape (tuple): Maximum shape of the dataset, this must be set if data is appended to the dataset.
        attributes (dict): Attributes to save.
        compression (str): Compression to use.
    """
    with h5py.File(file_path, "a") as file:
        if group_name not in file:
            raise Exception(f"Group {group_name} does not exist in the file.")

        group = file[group_name]

        if dataset_name in group:
            raise Exception(
                f"Dataset {dataset_name} already exists in the group {group_name}."
            )
        if not shape:
            dataset = group.create_dataset(
                dataset_name, data=data, compression=compression
            )
        else:
            maxshape = (None,) + shape[1:]
            print(maxshape)
            chunks = (1,) + shape[1:]
            print(chunks)
            dataset = group.create_dataset(
                dataset_name,
                data=data,
                maxshape=maxshape,
                chunks=chunks,
                compression=compression,
            )

        if attributes:
            for key, value in attributes.items():
                dataset.attrs[key] = value


def _parse_numpy_slicing(slicing_str: str) -> Tuple[slice, ...]:
    """
    Parse a numpy slicing string and convert it to a tuple of slice objects.

    Args:
        slicing_str (str): Numpy slicing string (e.g., "0:10, :, 50:100, :").

    Returns:
        Tuple[slice, ...]: A tuple of slice objects.
    """
    output = []
    slicing_str = slicing_str.replace(" ", "")
    slicing_str = slicing_str.replace("[", "")
    slicing_str = slicing_str.replace("]", "")
    slice_parts = slicing_str.split(",")
    for item in slice_parts:
        if item == ":":
            output.append(slice(None))
            continue
        if ":" in item:
            item = item.split(":")
            output.append(slice(int(item[0]), int(item[1])))
        else:
            output.append(int(item))
    return tuple(output)


def add_data_to_dataset(
    file_path: str, groupt_name: str, dataset_name: str, data: np.ndarray
) -> None:
    """
    Add data to a dataset in an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset.
        data (np.ndarray): Data to save.
        attributes (dict): Attributes to save.
        compression (str): Compression to use.
    """
    with h5py.File(file_path, "a") as file:
        full_dataset_path = f"{groupt_name}/{dataset_name}"
        if full_dataset_path not in file:
            raise Exception(f"Dataset {full_dataset_path} does not exist in the file.")
        dataset = file[full_dataset_path]
        dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
        dataset[-data.shape[0] :] = data
