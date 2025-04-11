"""
This module provides utilities for working with HDF5 files, including functions to display file structures,
retrieve and manipulate datasets, and create analysis files. It supports operations such as slicing datasets,
appending data, and managing attributes. The module leverages h5py for efficient HDF5 file handling.
"""

import os
from typing import Optional

import h5py
import numpy as np

from . import utils
from . import __version__


def display_file_structure(file_path: str) -> None:
    """
    Displays the structure (groups and datasets) of an HDF5 file.

    This function prints the hierarchy of groups and datasets in the specified HDF5 file,
    along with their shapes, data types, and attributes.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        None
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
    dataset_path: Optional[str] = None,
    slicing: Optional[str] = None,
) -> np.ndarray:
    """
    Retrieves data from a specified dataset in an HDF5 file.

    This function reads data from the specified dataset in the HDF5 file. Optionally,
    it applies slicing to retrieve a subset of the data.

    Args:
        path (str): Path to the HDF5 file or dataset.
        dataset_path (str, optional): Path to the dataset within the HDF5 file. If None,
                                      the dataset path is extracted from the input path.
        slicing (str, optional): String representation of slicing to apply to the dataset.

    Returns:
        np.ndarray: The retrieved data as a NumPy array.
    """
    if dataset_path is None:
        file_path, dataset_path = utils.split_h5_path(path)
    else:
        file_path = path
    if slicing is not None:
        slices = utils.parse_numpy_slicing(slicing)
    else:
        slices = None

    with h5py.File(file_path, "r") as file:
        dataset = file[dataset_path]
        assert isinstance(dataset, h5py.Dataset)
        if slices is not None:
            if dataset.ndim != len(slices):
                raise ValueError(
                    f"Dataset has {dataset.ndim} dimensions, but {len(slices)} slices were provided."
                ) from None
            data = dataset[tuple(slices)]
        else:
            data = dataset[:]

        data = data.astype(np.float64)
    return data


def add_array(
    path: str,
    data: np.ndarray,
    dataset_path: Optional[str] = None,
    attributes: Optional[dict] = None,
) -> None:
    """
    Adds an array to a dataset in an HDF5 file.

    If the dataset exists, the array is appended to it. If the dataset or its parent
    groups do not exist, they are created.

    Args:
        path (str): Path to the HDF5 file or dataset.
        data (np.ndarray): The data to add to the dataset.
        dataset_path (str, optional): Path to the dataset within the HDF5 file. If None,
                                      the dataset path is extracted from the input path.
        attributes (dict, optional): Attributes to add to the dataset.

    Returns:
        None
    """
    if dataset_path is None:
        file_path, dataset_path = utils.split_h5_path(path)
        parts = dataset_path.split("/")
    else:
        file_path = path
        parts = dataset_path.split("/")
    with h5py.File(file_path, "a", libver="latest") as file:
        # Split the dataset path into groups and dataset name
        groups = parts[:-1]
        dataset_name = parts[-1]
        # Create groups if they do not exist
        current_group = file
        for group in groups:
            assert isinstance(current_group, (h5py.File, h5py.Group))
            if group not in current_group.keys():
                current_group = current_group.create_group(group)
            else:
                current_group = current_group[group]
        # Check if the dataset already exists
        assert isinstance(current_group, (h5py.File, h5py.Group))
        if dataset_name not in current_group.keys():
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
        assert isinstance(current_dataset, h5py.Dataset)
        if current_dataset.shape[1:] != data.shape[1:]:
            raise ValueError(
                f"Shape of data to add ({data.shape[1:]}) does not match shape of existing dataset ({current_dataset.shape[1:]})"
            )
        current_dataset.resize(current_dataset.shape[0] + data.shape[0], axis=0)
        current_dataset[-data.shape[0] :] = data
        if attributes:
            for key, value in attributes.items():
                current_dataset.attrs[key] = value


def _get_params_from_data_file(file_path: str) -> tuple[int, int, int, int]:
    """
    Retrieves parameters from a data HDF5 file.

    This function extracts the total number of frames, column size, row size, and
    number of repetitions from the specified HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        tuple[int, int, int, int]: A tuple containing total_frames, column_size,
                                   row_size, and nreps.
    """
    with h5py.File(file_path, "r") as file:
        common_modes = file["preproc_common_modes"]
        assert isinstance(common_modes, h5py.Dataset)
        total_frames = common_modes.shape[0]
        row_size = common_modes.shape[1]
        nreps = common_modes.shape[2]
        preproc = file["preproc_mean_nreps"]
        assert isinstance(preproc, h5py.Dataset)
        column_size = preproc.shape[2]
    return total_frames, column_size, row_size, nreps


def _create_analysis_file(
    output_folder: str,
    output_filename: str,
    parameter_file_contents: dict,
    attributes_dict: dict,
) -> None:
    """
    Creates an analysis HDF5 file with predefined groups and metadata.

    This function creates an HDF5 file for analysis, including groups for offnoi,
    filter, and gain. It saves the parameter file contents and additional attributes
    as metadata.

    Args:
        output_folder (str): Path to the folder where the output file will be created.
        output_filename (str): Name of the output HDF5 file.
        parameter_file_contents (dict): Contents of the parameter file to save.
        attributes_dict (dict): Additional attributes to add to the HDF5 file.

    Returns:
        None
    """
    output_file = os.path.join(output_folder, output_filename)
    if os.path.exists(output_file):
        raise FileExistsError(f"File {output_file} already exists. Please delete")
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
