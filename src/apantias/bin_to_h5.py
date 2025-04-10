"""
This module provides functionality for processing binary data files and converting them into HDF5 format.
It includes utilities for workload distribution, data preprocessing, virtual dataset creation, and
frame-based statistical calculations. The module leverages multiprocessing for efficient handling of
large datasets and supports operations such as mean, median, and slope calculations.
"""

import multiprocessing
import time
import os
import gc

from typing import Optional
import h5py
import numpy as np

from . import utils
from . import __version__
from .logger import global_logger

_logger = global_logger


def _get_workload_dict(
    bin_files: list,
    h5_file_process: list,
    available_ram_gb: int,
    available_cpu_cores: int,
    row_size: int,
    key_ints: int,
    initial_offset: int,
) -> dict:
    """
    Calculates and returns a dictionary that defines the workload for each process when reading
    binary files and writing to HDF5 files.

    This function divides the binary files into batches that fit into the available RAM and assigns
    each batch to a process. Each process gets a portion of the file to read, along with an offset
    and the number of values to read.

    Args:
        bin_files (list): List of absolute paths to the binary files.
        h5_file_process (list): List of absolute paths to the HDF5 files for each process.
        available_ram_gb (int): Available RAM per process in gigabytes.
        available_cpu_cores (int): Number of available CPU cores.
        row_size (int): Size of a row in bytes.
        key_ints (int): Number of key integers.
        initial_offset (int): Offset in bytes to start reading.

    Returns:
        dict: A dictionary with the workload for each process.

    Example:
        Output dictionary format:
        {
            'bin_file1': [
                [
                    [offset1, counts1, group1],
                    [offset2, counts2, group2],
                    ...
                ],  # Batch 1 for process 0
                [
                    [offset1, counts1, group1],
                    [offset2, counts2, group2],
                    ...
                ],  # Batch 2 for process 1, if needed
                ...
            ],
            'bin_file2': [
                [
                    [offset1, counts1, group1],
                    [offset2, counts2, group2],
                    ...
                ],
                ...
            ],
            ...
        }
    """
    # 10% of available ram will be used to load raw_data from the bin_file file.
    # it is converted from uint16 to float64, which is 4 times the size, the rest is needed
    # for computation.
    # This value needs to be adjusted if the multicore processes need more ram for some reason.
    # Example: np.nanmean would need double the ram as np.mean!
    available_ram = int((available_ram_gb * 1024 * 1024 * 1024) * 0.1)
    available_ram_per_process = int(available_ram / available_cpu_cores)
    rows_read_per_process = int(available_ram_per_process / ((row_size + key_ints) * 2))
    workload_dict = {}
    for index, bin_file in enumerate(bin_files):
        bin_list = []
        # bin_size is in unit bytes
        bin_size = os.path.getsize(bin_file)
        bin_name = os.path.basename(bin_file).split(".")[0].split(".")[0]
        # check how often all subprocesses must be called to read the whole file
        batches = int(bin_size / available_ram) + 1
        # the offset is named that way because of the kwarg of np.fromfile
        current_offset = initial_offset
        for i in range(batches):
            bin_list.append([])
            bytes_left = bin_size - current_offset
            if bytes_left > available_ram:
                for n in range(available_cpu_cores):
                    # counts is the number of uint16 values to read
                    counts = rows_read_per_process * (row_size + key_ints)
                    group = f"{h5_file_process[n]}/{index}_{bin_name}/batch_{i}/"
                    bin_list[i].append([current_offset, counts, group])
                    # set the new offset in bytes
                    current_offset += counts * 2
            else:
                bytes_left_per_process = int(bytes_left / available_cpu_cores)
                rows_left_read_per_process = int(bytes_left_per_process / ((row_size + key_ints) * 2))
                for n in range(available_cpu_cores):
                    # counts is the number of uint16 values to read
                    counts = rows_left_read_per_process * (row_size + key_ints)
                    group = f"{h5_file_process[n]}/{index}_{bin_name}/batch_{i}/"
                    bin_list[i].append([current_offset, counts, group])
                    # set the new offset in bytes
                    current_offset += counts * 2
        workload_dict[bin_file] = bin_list
    # No complete frames can be found if there is too little data left in the last batch
    # Therefore, the last batch is removed if it contains less than 50% of data
    for bin_file, workload in workload_dict.items():
        first_count = workload[0][0][1]
        last_count = workload[-1][0][1]
        if last_count / first_count < 0.5:
            # remove the last batch
            workload.pop()
    return workload_dict


def _get_vds_list(workload_dict: dict, old_list: list = None) -> list:  # type: ignore
    """
    Creates a list used for the creation of virtual datasets (VDS).

    This function generates a list of datasets present in the HDF5 files of the processes.
    It calculates the final shape of each dataset (sum of all shapes across axis 0) and
    collects the sources of the datasets. If an old list of datasets is provided, the
    datasets in the old list are ignored and not added to the new list.

    Args:
        workload_dict (list): A dictionary defining the workload for each process,
                              including the HDF5 groups and datasets.
        old_list (list, optional): A list of datasets to ignore when creating the new list.

    Returns:
        list: A list of dictionaries, where each dictionary represents a dataset with
              its name, final shape, sources, and attributes.

    Example:
        Output list format:
        [
            {
                "name": "common_modes",
                "final_shape": [1000, 64, 200, 1],
                "sources": [
                    "path/to/file1.h5/group1/dataset1",
                    "path/to/file2.h5/group2/dataset2",
                    ...
                ],
                "attributes": {"attr1": "value1", "attr2": "value2"}
            },
            {
                "name": "mean_frames",
                "final_shape": [1000, 64, 200],
                "sources": [
                    "path/to/file1.h5/group1/dataset3",
                    "path/to/file2.h5/group2/dataset4",
                    ...
                ],
                "attributes": {"attr1": "value1", "attr2": "value2"}
            },
            ...
        ]
    """
    if old_list is None:
        old_list = []
    datasets = []
    for bin_file in workload_dict.keys():
        for _, batch in enumerate(workload_dict[bin_file]):
            for _, [_, _, h5_group] in enumerate(batch):
                # get names and shapes of datasets in the group
                new_datasets = _get_datasets_from_h5(h5_group)
                if not datasets:
                    datasets = new_datasets
                else:
                    for i, new_dataset in enumerate(new_datasets):
                        new_name = new_dataset[0]
                        new_shape_frames = new_dataset[1][0]
                        if new_name not in [dataset[0] for dataset in datasets]:
                            raise ValueError(f"Dataset {new_name} not found in datasets")
                        # increment the shape of the dataset by the new shape across axis 0
                        datasets[i][1][0] += new_shape_frames
    # the datsets_dict contains the names of all datasets present in the .h5 files of the processes,
    # the final shape (sum of all shapes across axis 0) and the sources of the datasets
    vds_list = []
    for dataset in datasets:
        dataset_name = dataset[0]
        if old_list:
            # check if datset_name is in the old_list
            if not any(d.get("name") == dataset_name for d in old_list):
                vds_list.append(
                    {
                        "name": dataset_name,
                        "final_shape": dataset[1],
                        "sources": [],
                        "attributes": dataset[2],
                    }
                )
        else:
            vds_list.append(
                {
                    "name": dataset_name,
                    "final_shape": dataset[1],
                    "sources": [],
                    "attributes": dataset[2],
                }
            )
    # add the sources
    for bin_file in workload_dict.keys():
        for _, batch in enumerate(workload_dict[bin_file]):
            for _, [_, _, h5_group] in enumerate(batch):
                for dataset_info in vds_list:
                    name = dataset_info["name"]
                    dataset_info["sources"].append(f"{h5_group}{name}")
    return vds_list


def _avg_frames(h5_file: str, vds_list: list):
    """
    Calculates averages over frames for datasets in the virtual dataset list.

    Args:
        h5_file (str): Path to the HDF5 file.
        vds_list (list): List of datasets with attributes specifying the type of averaging.

    Returns:
        None
    """
    for dataset in vds_list:
        with h5py.File(h5_file, "a") as f:
            name = dataset["name"]
            avg = dataset["attributes"]["avg"]
            # skip raw_data, loading it would take too much ram in most instances
            if avg == "False":
                continue
            if "slice" in name:
                group_name = name.split("_", 1)[0]
                dset_name = name.split("_", 1)[1]
                name = f"{group_name}/{dset_name}"
            source = np.array(f[name])
            if avg == "sum":
                summed = np.sum(source, axis=0)
                f.create_dataset(name + "_sum_frames", data=summed)
            elif avg == "mean":
                average = utils.nanmean(source, axis=0)
                f.create_dataset(name + "_mean_frames", data=average)
            elif avg == "median":
                median = np.median(source, axis=0)
                f.create_dataset(name + "_median_frames", data=median)
            elif avg == "weighted":
                dset = f["raw_data"]
                assert isinstance(dset, h5py.Dataset)
                total_frames = dset.shape[0]
                weighted_avg = np.sum(source, axis=0) / total_frames
                f.create_dataset(name + "_weighted_frames", data=weighted_avg)


def _create_vds(h5_file: str, vds_list: list):
    """
    Creates virtual datasets in an HDF5 file based on the provided dataset list.

    Args:
        h5_file (str): Path to the HDF5 file.
        vds_list (list): List of datasets with their sources and attributes.

    Returns:
        None
    """
    for dataset in vds_list:
        name = dataset["name"]
        sources = dataset["sources"]
        final_shape = tuple(dataset["final_shape"])
        # get type of first dataset
        dset = h5py.File(sources[0].split(".h5")[0] + ".h5", "r")[sources[0].split(".h5")[1]]
        assert isinstance(dset, h5py.Dataset)
        layout = h5py.VirtualLayout(shape=final_shape, dtype=dset.dtype)
        with h5py.File(h5_file, "a") as f:
            start_index = 0
            for source in sources:
                source_h5 = source.split(".h5")[0] + ".h5"
                source_dataset = source.split(".h5")[1]
                with h5py.File(source_h5, "r") as source_f:
                    dset = source_f[source_dataset]
                    assert isinstance(dset, h5py.Dataset)
                    sh = dset.shape
                    attributes = dict(dset.attrs)
                end_index = start_index + sh[0]
                vsource = h5py.VirtualSource(source_h5, source_dataset, shape=sh)
                layout[start_index:end_index, ...] = vsource
                start_index = end_index
            # fillvalue = np.nan means, that if the source dataset is not present, the value is np.nan
            # so if the absolute path to the source file changes, the value will be np.nan
            if "slice" in name:
                group_name = name.split("_", 1)[0]
                dset_name = name.split("_", 1)[1]
                if group_name not in f:
                    group = f.create_group(group_name)
                else:
                    group = f[group_name]
                assert isinstance(group, h5py.Group)
                dset = group.create_virtual_dataset(dset_name, layout, fillvalue=np.nan)
                for key, value in attributes.items():
                    dset.attrs[key] = value
            else:
                dset = f.create_virtual_dataset(name, layout, fillvalue=np.nan)
                for key, value in attributes.items():
                    dset.attrs[key] = value


def _read_data_from_bin(
    bin_file: str,
    column_size: int,
    row_size: int,
    key_ints: int,
    nreps: int,
    offset: int,
    counts: int,
) -> np.ndarray:
    """
    Reads and reshapes data from a binary file.

    Args:
        bin_file (str): Path to the binary file.
        column_size (int): Number of columns in the binary data.
        row_size (int): Number of rows in the binary data.
        key_ints (int): Number of key integers in the binary data.
        nreps (int): Number of repetitions in the binary data.
        offset (int): Offset in bytes to start reading.
        counts (int): Number of uint16 values to read.

    Returns:
        np.ndarray: Reshaped data from the binary file.
    """
    raw_row_size = row_size + key_ints
    rows_per_frame = column_size * nreps
    # count parameter needs to be in units of uint16 (uint16 = 2 bytes)
    inp_data = np.fromfile(bin_file, dtype="uint16", count=counts, offset=offset)
    # check if file is at its end
    if inp_data.size == 0:
        raise ValueError(f"File {bin_file} is empty or corrupted")
    # reshape the array into rows -> (#ofRows,67)
    try:
        inp_data = inp_data.reshape(-1, raw_row_size)
    except ValueError as exc:
        raise ValueError(f"Could not reshape data from {bin_file}. Check if the file is corrupted.") from exc
    # find all the framekeys
    frame_keys = np.where(inp_data[:, column_size] == 65535)
    # stack them and calculate difference to find incomplete frames
    frames = np.stack((frame_keys[0][:-1], frame_keys[0][1:]))
    diff = np.diff(frames, axis=0)
    valid_frames_position = np.nonzero(diff == rows_per_frame)[1]
    if len(valid_frames_position) == 0:
        raise ValueError("No valid frames found")
    valid_frames = frames.T[valid_frames_position]
    frame_start_indices = valid_frames[:, 0]
    frame_end_indices = valid_frames[:, 1]
    inp_data = np.array(
        [inp_data[start + 1 : end + 1, :64] for start, end in zip(frame_start_indices, frame_end_indices)]
    )
    inp_data = inp_data.reshape((-1, column_size, nreps, row_size))
    return inp_data


def _write_data_to_h5(path: str, data: np.ndarray, attributes=None) -> None:
    """
    Writes data to a specified dataset in an HDF5 file.

    Args:
        path (str): Full path to the dataset in the HDF5 file.
        data (np.ndarray): The data to write.
        attributes (dict, optional): Attributes to add to the dataset.

    Returns:
        None
    """
    h5_file, dataset_path = utils.split_h5_path(path)
    with h5py.File(h5_file, "a") as f:
        path_parts = dataset_path.strip("/").split("/")
        groups = path_parts[:-1]
        dataset = path_parts[-1]
        current_group = f
        for group in groups:
            assert isinstance(current_group, (h5py.Group, h5py.File))
            if group not in current_group.keys():
                current_group = current_group.create_group(group)
            else:
                current_group = current_group[group]
        assert isinstance(current_group, (h5py.Group, h5py.File))
        if dataset in current_group.keys():
            raise ValueError(f"Dataset {dataset} already exists in {h5_file}")
        dataset = current_group.create_dataset(dataset, dtype=data.dtype, data=data, chunks=None)
        if attributes is not None:
            for key, value in attributes.items():
                dataset.attrs[key] = value


def _read_data_from_h5(path: str) -> np.ndarray:
    """
    Reads data from a specified dataset in an HDF5 file.

    Args:
        path (str): Full path to the dataset in the HDF5 file.

    Returns:
        np.ndarray: The data read from the dataset.
    """
    h5_file, dataset_path = utils.split_h5_path(path)
    with h5py.File(h5_file, "r") as f:
        data = f[dataset_path]
        assert isinstance(data, h5py.Dataset)
        data = data[:]
    return data


def _get_datasets_from_h5(path: str) -> list:
    """
    Retrieves a list of datasets in a specified HDF5 group.

    Args:
        path (str): Full path to the HDF5 group.

    Returns:
        list: A list of datasets, including their names, shapes, and attributes.
    """
    h5_file, group_path = utils.split_h5_path(path)
    with h5py.File(h5_file, "r") as f:
        datasets = []
        group = f[group_path]
        assert isinstance(group, (h5py.File, h5py.Group))
        for name, item in group.items():
            assert isinstance(item, h5py.Dataset)
            datasets.append([name, list(item.shape), dict(item.attrs)])
    return datasets


def _process_raw_data(
    h5_group: str,
    column_size: int,
    row_size: int,
    key_ints: int,
    ignore_first_nreps: int,
    nreps: int,
    offset: int,
    counts: int,
    bin_file: str,
    polarity: int,
) -> None:
    """
    Processes raw binary data and writes it to an HDF5 file.

    This function reads raw data from a binary file, applies basic processing such as ignoring
    initial repetitions, and calculates statistical metrics like mean, median, and standard
    deviation. The processed data and metrics are written to the specified HDF5 group.

    Args:
        h5_group (str): Path to the HDF5 group where the processed data will be stored.
        column_size (int): Number of columns in the binary data.
        row_size (int): Number of rows in the binary data.
        key_ints (int): Number of key integers in the binary data.
        ignore_first_nreps (int): Number of initial repetitions to ignore during processing.
        nreps (int): Number of repetitions in the binary data.
        offset (int): Offset in bytes to start reading the binary file.
        counts (int): Number of uint16 values to read from the binary file.
        bin_file (str): Path to the binary file to process.
        polarity: default is -1. raw data is multiplied by this value.

    Returns:
        None
    """
    # read data from bin_file file, multiple processes can read from the same file
    # write the avg attribute to the dset to determine what to average later in the vds
    try:
        data = _read_data_from_bin(bin_file, column_size, row_size, key_ints, nreps, offset, counts)
        # data is saved to file in uint16 to save space
        _write_data_to_h5(h5_group + "raw_data", data, {"avg": "False"})
        data = data[:, :, ignore_first_nreps:, :]
        # performing the mean automatically casts to float64, multiply by polarity now
        raw_offset = np.mean(data, axis=0, keepdims=True) * data.shape[0] * polarity
        raw_data_mean = np.mean(data, axis=2) * polarity
        raw_data_median = np.median(data, axis=2) * polarity
        raw_data_std = np.std(data, axis=2) * polarity
        _write_data_to_h5(h5_group + "raw_offset", raw_offset, {"avg": "weighted"})
        _write_data_to_h5(h5_group + "raw_data_mean_nreps", raw_data_mean, {"avg": "mean"})
        _write_data_to_h5(h5_group + "raw_data_std_nreps", raw_data_std, {"avg": "mean"})
        _write_data_to_h5(h5_group + "raw_data_median_nreps", raw_data_median, {"avg": "mean"})
        del raw_offset, raw_data_mean, raw_data_median, raw_data_std, data
        gc.collect()
    except Exception as e:
        raise e
    finally:
        gc.collect()


def _preprocess(
    h5_group: str,
    h5_file_virtual: str,
    ignore_first_nreps: int,
    ext_dark_frame_dset: str,
    offset: int,
    nreps_eval: list,
    polarity: int,
) -> None:
    """
    Preprocesses raw data by applying corrections and calculating statistical metrics.

    This function reads raw data from an HDF5 file, applies corrections such as subtracting
    offsets and common modes, and calculates statistical metrics like mean, median, standard
    deviation, and slopes. The results are written back to the HDF5 file. Optionally, it can
    process specific evaluation ranges for repetitions.

    Args:
        h5_group (str): Path to the HDF5 group containing the raw data.
        h5_file_virtual (str): Path to the virtual HDF5 file.
        ignore_first_nreps (int): Number of initial repetitions to ignore during processing.
        ext_dark_frame_dset (str): Path to an external dark frame dataset (optional).
        offset (int): Offset value for data correction.
        nreps_eval (list): List of evaluation ranges for repetitions (optional).
        polarity: default is -1. raw data is multiplied by this value.

    Returns:
        None
    """
    try:
        data = _read_data_from_h5(h5_group + "raw_data")
        data = data[:, :, ignore_first_nreps:, :]
        data = data.astype(np.float64) * polarity
        if ext_dark_frame_dset is not None:
            offset_map = _read_data_from_h5(ext_dark_frame_dset)
        else:
            offset_map = _read_data_from_h5(h5_file_virtual + "raw_offset_weighted_frames")
        data -= offset_map
        common_modes = np.median(data, axis=3, keepdims=True)
        data -= common_modes
        _write_data_to_h5(h5_group + "preproc_common_modes", common_modes, {"avg": "mean"})
        del offset, common_modes
        gc.collect()
        if data.shape[2] > 50:
            shapiro = utils.shapiro(data, axis=2)
            _write_data_to_h5(h5_group + "preproc_shapiro", shapiro, {"avg": "mean"})
            del shapiro
        mean = np.mean(data, axis=2)
        std = np.std(data, axis=2)
        median = np.median(data, axis=2)
        slopes = utils.apply_slope_fit_along_frames_single(data)
        _write_data_to_h5(h5_group + "preproc_mean_nreps", mean, {"avg": "mean"})
        _write_data_to_h5(h5_group + "preproc_median_nreps", median, {"avg": "mean"})
        _write_data_to_h5(h5_group + "preproc_std_nreps", std, {"avg": "mean"})
        _write_data_to_h5(h5_group + "preproc_slope_nreps", slopes, {"avg": "mean"})
        del mean, std, median, slopes
        gc.collect()
        if nreps_eval is None:
            return
        for item in nreps_eval:
            s = slice(item[0], item[1], item[2])
            data_slice = data[:, :, s, :]
            if data_slice.shape[2] > 50:
                shapiro = utils.shapiro(data_slice, axis=2)
                _write_data_to_h5(h5_group + f"{s}_preproc_shapiro", shapiro, {"avg": "mean"})
                del shapiro
            mean = np.mean(data_slice, axis=2)
            std = np.std(data_slice, axis=2)
            median = np.median(data_slice, axis=2)
            x = np.arange(data_slice.shape[2])
            slopes = np.apply_along_axis(lambda y, x=x: np.polyfit(x, y, 1)[0], axis=2, arr=data_slice)
            _write_data_to_h5(h5_group + f"{s}_preproc_mean_nreps", mean, {"avg": "mean"})
            _write_data_to_h5(h5_group + f"{s}_preproc_median_nreps", median, {"avg": "mean"})
            _write_data_to_h5(h5_group + f"{s}_preproc_std_nreps", std, {"avg": "mean"})
            _write_data_to_h5(h5_group + f"{s}_preproc_slope_nreps", slopes, {"avg": "mean"})
            del data_slice, mean, std, median, slopes
            gc.collect()
        del data
        gc.collect()

    except Exception as e:
        raise e
    finally:
        gc.collect()


def create_data_file_from_bins(
    bin_files: list[str],
    output_folder: str,
    column_size: int = 64,
    row_size: int = 64,
    key_ints: int = 3,
    ignore_first_nreps: int = 3,
    offset: int = 8,
    available_cpu_cores: int = 4,
    available_ram_gb: int = 16,
    ext_dark_frame_h5: Optional[str] = None,
    nreps_eval: Optional[list[list[int]]] = None,
    attributes: Optional[dict] = None,
    polarity: int = -1,
) -> None:
    """
    Processes binary data files and converts them into HDF5 format with virtual datasets.

    This function reads binary files, splits the data into manageable batches, and processes
    them in parallel using multiple CPU cores. It creates HDF5 files for each process,
    generates virtual datasets for easy access, and calculates averages over frames.
    The resulting HDF5 file contains the processed data and metadata.

    Args:
        bin_files (list[str]): List of paths to the binary files to process.
        output_folder (str): Path to the folder where the output HDF5 files will be stored.
        column_size (int): Number of columns in the binary data (default: 64).
        row_size (int): Number of rows in the binary data (default: 64).
        key_ints (int): Number of key integers in the binary data (default: 3).
        ignore_first_nreps (int): Number of repetitions to ignore during processing (default: 3).
        offset (int): Offset in bytes to start reading the binary files (default: 8).
        available_cpu_cores (int): Number of CPU cores to use for parallel processing (default: 4).
        available_ram_gb (int): Amount of available RAM in gigabytes (default: 16).
        ext_dark_frame_h5 (str): Path to an external dark frame HDF5 file (optional).
        nreps_eval (list[list[int]]): List of evaluation ranges for repetitions (optional).
        attributes (dict): Additional attributes to add to the HDF5 files (optional).
        polarity: default is -1. raw data is multiplied by this value.

    Raises:
        FileNotFoundError: If any of the specified files or folders do not exist.
        ValueError: If the binary files have inconsistent data or invalid parameters.

    Returns:
        None
    """
    # check if folder, bin_file files exist and calculate nreps and make sure they are all the same
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"Folder {output_folder} does not exist.")
    nreps_list = []
    for bin_file in bin_files:
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"File {bin_file} does not exist")
        else:
            raw_row_size = row_size + key_ints
            test_data = np.fromfile(
                bin_file,
                dtype="uint16",
                # load some frames,
                count=column_size * raw_row_size * 1000 * 40,
                offset=8,
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
            nreps_list.append(estimated_nreps)
    # check if all nreps are the same
    if all(x == nreps_list[0] for x in nreps_list):
        nreps = nreps_list[0]
    else:
        raise ValueError(f"Not all bin_file files have the same number of nreps: {nreps_list}")
    # check if external dark frame exists and has the right shape
    if ext_dark_frame_h5 is not None:
        ext_h5_file = ext_dark_frame_h5.split(".h5")[0] + ".h5"
        ext_group_path = ext_dark_frame_h5.split(".h5")[1]
        if not os.path.exists(ext_h5_file):
            raise FileNotFoundError(f'File "{ext_h5_file}" does not exist')
        with h5py.File(ext_h5_file, "r") as f:
            dset = f[ext_group_path]
            assert isinstance(dset, h5py.Dataset)
            if dset.shape[0] != column_size or dset.shape[1] != row_size:
                raise ValueError(
                    f"Shape of external dark frame {ext_dark_frame_h5} does"
                    "not match ({column_size}, {row_size}) of the bin_file files"
                )
    # create folders:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    bin_name = os.path.basename(bin_files[0]).split(".")[0]
    working_folder = os.path.join(output_folder, f"{timestamp}_{bin_name}")
    data_folder = os.path.join(working_folder, "data")
    os.mkdir(working_folder)
    os.mkdir(data_folder)
    # create a h5 file for every process in the data_folder
    h5_file_process = [os.path.join(data_folder, f"data_{i}.h5") for i in range(available_cpu_cores)]

    for file in h5_file_process:
        with h5py.File(file, "w") as f:
            f.attrs["info"] = (
                "This file contains data from one subprocess to enable parallel"
                "processing. Retrieve the whole measurement data from the virtual dataset in the"
                f"folder {working_folder}."
            )
            f.attrs["apantias-version"] = __version__
    # and one for the virtual datasets
    h5_file_virtual = os.path.join(working_folder, f"{bin_name}.h5")
    with h5py.File(h5_file_virtual, "w") as f:
        f.attrs["info"] = (
            "This file contains virtual datasets to access the data from the "
            f"folder {data_folder}. You can move this file, but the data in it"
            "is linked to the data folder. If the folder is moved there will be"
            "no data available in the file."
        )
        f.attrs["apantias-version"] = __version__
        if attributes is not None:
            for key, value in attributes.items():
                f.attrs[key] = value
        if ext_dark_frame_h5 is None:
            f.attrs["ext_dark_frame_offset"] = "None"
        else:
            f.attrs["ext_dark_frame_offset"] = ext_dark_frame_h5

    # get the workload dictionary for each process
    workload_dict = _get_workload_dict(
        bin_files,
        h5_file_process,
        available_ram_gb,
        available_cpu_cores,
        row_size,
        key_ints,
        offset,
    )
    _logger.info("Starting preprocessing step.")
    _logger.info("These .bin_file files will be processed:")
    for bin_file in bin_files:
        _logger.info("%s of size %.2f GB", bin_file, os.path.getsize(bin_file) / (1024**3))
        _logger.info(
            "The file will be split into %d batches to fit into memory.",
            len(workload_dict[bin_file]),
        )
    _logger.info("Note, that the provided bin_file files will be treated as being from the same measurement.")
    _logger.info("If you wish to process multiple measurements, please provide them separately.")
    _logger.info("Start processing Raw Data.")
    for bin_file, workload in workload_dict.items():
        _logger.info("Start processing %s", bin_file)
        for batch_index, batch in enumerate(workload):
            processes = []
            for _, [offset_batch, counts, h5_group] in enumerate(batch):
                p = multiprocessing.Process(
                    target=_process_raw_data,
                    args=(
                        h5_group,
                        column_size,
                        row_size,
                        key_ints,
                        ignore_first_nreps,
                        nreps,
                        offset_batch,
                        counts,
                        bin_file,
                        polarity,
                    ),
                )
                processes.append(p)
                p.start()
            _logger.info(
                "batch %d/%d started, %d processes are running.",
                batch_index + 1,
                len(workload_dict[bin_file]),
                available_cpu_cores,
            )
            finished = []
            while any(p.is_alive() for p in processes):
                finished_new = [i for i, p in enumerate(processes) if not p.is_alive]
                if not finished == finished_new:
                    _logger.info("Processes finished so far: %s", finished_new)
                    finished = finished_new
                time.sleep(5)
            for p in processes:
                p.join()
    _logger.info("Raw Data processed.")
    vds_list = _get_vds_list(workload_dict)
    _logger.info("Creating virtual dataset.")
    _create_vds(h5_file_virtual, vds_list)
    _logger.info("Virtual dataset created.")
    _logger.info("Calculating averages over frames.")
    _avg_frames(h5_file_virtual, vds_list)
    _logger.info("Averages over frames calculated.")
    _logger.info("Processing of Raw Data finished.")
    _logger.info("Start preprocessing.")
    for bin_file, workload in workload_dict.items():
        _logger.info("Start processing %s", bin_file)
        for batch_index, batch in enumerate(workload):
            processes = []
            for _, [offset_batch, counts, h5_group] in enumerate(batch):
                p = multiprocessing.Process(
                    target=_preprocess,
                    args=(
                        h5_group,
                        h5_file_virtual,
                        ignore_first_nreps,
                        ext_dark_frame_h5,
                        offset_batch,
                        nreps_eval,
                        polarity,
                    ),
                )
                processes.append(p)
                p.start()
            _logger.info(
                "batch %d/%d started, %d processes are running.",
                batch_index + 1,
                len(workload_dict[bin_file]),
                available_cpu_cores,
            )
            finished = []
            while any(p.is_alive() for p in processes):
                finished_new = [i for i, p in enumerate(processes) if not p.is_alive]
                if not finished == finished_new:
                    _logger.info("Processes finished so far: %s", finished_new)
                    finished = finished_new
                time.sleep(5)
            for p in processes:
                p.join()
    _logger.info("Preprocessing finished.")
    new_vds_list = _get_vds_list(workload_dict, vds_list)
    _logger.info("Creating virtual dataset.")
    _create_vds(h5_file_virtual, new_vds_list)
    _logger.info("Virtual dataset created.")
    _logger.info("Calculating averages over frames.")
    _avg_frames(h5_file_virtual, new_vds_list)
    _logger.info("Averages over frames calculated.")
    _logger.info("Final Dataset is stored in %s", h5_file_virtual)
    _logger.info("Data is stored in %s", data_folder)
    _logger.info("DO NOT move the data folder, if you do the virtual dataset will not work anymore.")
    _logger.info("You can move the .h5 file to any location.\n")
