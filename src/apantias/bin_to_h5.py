import h5py
import numpy as np
import multiprocessing
import time
from typing import List, Tuple, Optional
import os
import gc

from . import file_io as io
from . import utils
from . import __version__
from .logger import global_logger

_logger = global_logger


def _get_workload_dict(
    bin_files: List,
    h5_file_process: List,
    available_ram_gb: int,
    available_cpu_cores: int,
    row_size: int,
    key_ints: int,
    initial_offset: int,
) -> dict:
    """
    calculates and returns a dictionary that defines the workload for each process when reading
    binary files and writing to HDF5 files. It divides the binary files into batches that fit into
    the available RAM and assigns each batch to a process. Each process gets a portion of the file
    to read, along with an offset and the number of values to read.

    {
    'bin_file1': [
        [(offset1, counts1, group1), (offset2, counts2, group2), ...],  # Batch 1 for process 0
        [(offset1, counts1, group1), (offset2, counts2, group2), ...],  # Batch 2 for process 1, if needed
        ...
    ],
    'bin_file2': [
        [(offset1, counts1, group1), (offset2, counts2, group2), ...],
        [(offset1, counts1, group1), (offset2, counts2, group2), ...],
        ...
    ],
    ...
    }
    Args:
        bin_files: List of absolute paths to the binary files.
        h5_file_process: List of absolute paths to the HDF5 files for each process.
        available_ram_gb: Available RAM per process in gigabytes.
        available_cpu_cores: Number of available CPU cores.
        row_size: Size of a row in bytes.
        key_ints: Number of key integers.
        initial_offset: Offset in bytes to start reading.
    Returns:
        workload_dict: dictionary with the workload for each process
    """
    # 10% of available ram will be used to load raw_data from the bin file.
    # it is converted from uint16 to float64, which is 4 times the size
    # Note: when using np.mean (or np.median) instead of np.mean this value can be doubled.
    available_ram = int((available_ram_gb * 1024 * 1024 * 1024) * 0.1)
    available_ram_per_process = int(available_ram / available_cpu_cores)
    rows_read_per_process = int(available_ram_per_process / ((row_size + key_ints) * 2))
    workload_dict = {}
    for index, bin in enumerate(bin_files):
        bin_list = []
        # bin_size is in unit bytes
        bin_size = os.path.getsize(bin)
        bin_name = os.path.basename(bin).split(".")[0].split(".")[0]
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
                rows_left_read_per_process = int(
                    bytes_left_per_process / ((row_size + key_ints) * 2)
                )
                for n in range(available_cpu_cores):
                    # counts is the number of uint16 values to read
                    counts = rows_left_read_per_process * (row_size + key_ints)
                    group = f"{h5_file_process[n]}/{index}_{bin_name}/batch_{i}/"
                    bin_list[i].append([current_offset, counts, group])
                    # set the new offset in bytes
                    current_offset += counts * 2
        workload_dict[bin] = bin_list
    #No complete frames can be found if there is too little data left in the last batch
    #Therefore, the last batch is removed if it contains less than 50% of data
    for bin in workload_dict.keys():
        first_count = workload_dict[bin][0][0][1]
        last_count = workload_dict[bin][-1][0][1]
        if last_count/first_count < 0.5:
            #remove the last batch
            workload_dict[bin].pop()

    return workload_dict


def _get_vds_list(workload_dict: List, old_list: List = None) -> List:
    """
    Creates a list used for the creation of virtual datasets. (vds)
    The vds_dict contains the names of all datasets present in the .h5 files of the processes,
    the final shape (sum of all shapes across axis 0) and the sources of the datasets.
    Example:
    [
        "name" : "common_modes",
        "final_shape" : [1000,64,200,1],
        "sources" : ["path/file.h5/group/dataset_name",...]}
        ...
    ]
    If old_dict is provided, the datasets in old_dict are ignored and not added to the new dict.
    """
    datasets = []
    for bin in workload_dict.keys():
        for batch_index, batch in enumerate(workload_dict[bin]):
            for process_index, [offset, counts, h5_group] in enumerate(batch):
                # get names and shapes of datasets in the group
                new_datasets = _get_datasets_from_h5(h5_group)
                if datasets == []:
                    datasets = new_datasets
                else:
                    for i, new_dataset in enumerate(new_datasets):
                        new_name = new_dataset[0]
                        new_shape_frames = new_dataset[1][0]
                        if new_name not in [dataset[0] for dataset in datasets]:
                            raise ValueError(
                                f"Dataset {new_name} not found in datasets"
                            )
                        else:
                            # increment the shape of the dataset by the new shape across axis 0
                            datasets[i][1][0] += new_shape_frames
    # the datsets_dict contains the names of all datasets present in the .h5 files of the processes,
    # the final shape (sum of all shapes across axis 0) and the sources of the datasets
    vds_list = []
    for dataset in datasets:
        dataset_name = dataset[0]
        if old_list is not None:
            # check if datset_name is in the old_list
            if not any(d.get("name") == dataset_name for d in old_list):
                vds_list.append(
                    {
                        "name": dataset_name,
                        "final_shape": dataset[1],
                        "sources": [],
                        "attributes": dataset[2]
                    }
                )
        else:
            vds_list.append(
                {
                    "name": dataset_name,
                    "final_shape": dataset[1],
                    "sources": [],
                    "attributes": dataset[2]
                }
            )
    # add the sources
    for bin in workload_dict.keys():
        for batch_index, batch in enumerate(workload_dict[bin]):
            for process_index, [offset, counts, h5_group] in enumerate(batch):
                for dataset_info in vds_list:
                    name = dataset_info["name"]
                    dataset_info["sources"].append(f"{h5_group}{name}")
    return vds_list


def _avg_frames(h5_file, vds_list):
    """
    Creates averages over frames for all datasets in the vds_dict.
    The information which calculation to perform is in the attributes of the dset.
    "avg" : mean/median/std/sum/weighted
    "axis" : axis over which mean/median/std/sum
    "weight" : weighted avg is always over frames, so axis is not needed
    "group" : if it should be saved in a separate group in the virtualdset
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
            if avg =="sum":
                sum = np.sum(source, axis=0)
                f.create_dataset(name + "_sum_frames", data=sum)
            elif avg == "mean":
                average = utils.nanmean(source, axis=0)
                f.create_dataset(name + "_mean_frames", data=average)
            elif avg == "median":
                median = np.median(source, axis=0)
                f.create_dataset(name + "_median_frames", data=median)	
            elif avg == "weighted":
                total_frames = f["raw_data"].shape[0]
                weighted_avg = np.sum(source, axis=0)/total_frames
                f.create_dataset(name + "_weighted_frames", data=weighted_avg)


def _create_vds(h5_file, vds_list):
    """
    Iterates through the datasets in the vds_dict and creates virtual datasets in the h5 file.
    """
    for dataset in vds_list:
        name = dataset["name"]
        sources = dataset["sources"]
        final_shape = tuple(dataset["final_shape"])
        # get type of first dataset
        dtype = h5py.File(sources[0].split(".h5")[0] + ".h5", "r")[
            sources[0].split(".h5")[1]
        ].dtype
        layout = h5py.VirtualLayout(shape=final_shape, dtype=dtype)
        with h5py.File(h5_file, "a") as f:
            start_index = 0
            for source in sources:
                source_h5 = source.split(".h5")[0] + ".h5"
                source_dataset = source.split(".h5")[1]
                with h5py.File(source_h5, "r") as source_f:
                    dset = source_f[source_dataset]
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
    Reads an reshapes data from a binary file.
    Args:
        bin_file: absolute path to the binary file
        column_size: number of columns in the binary file
        row_size: number of rows in the binary file
        key_ints: number of key integers
        nreps: number of repetitions
        offset: offset in bytes to start reading
        counts: number of uint16 values to read
    Returns:
        inp_data: reshaped data from the binary file
    """
    raw_row_size = row_size + key_ints
    rows_per_frame = column_size * nreps
    chunk_size = counts * 2
    # count parameter needs to be in units of uint16 (uint16 = 2 bytes)
    inp_data = np.fromfile(bin_file, dtype="uint16", count=counts, offset=offset)
    offset += chunk_size  # offset is in bytes, uint16 = 16 bit = 2 bytes
    # check if file is at its end
    if inp_data.size == 0:
        return None
    # reshape the array into rows -> (#ofRows,67)
    try:
        inp_data = inp_data.reshape(-1, raw_row_size)
    except ValueError:
        raise ValueError(
            f"Could not reshape data from {bin_file}. Check if the file is corrupted."
        )
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
        [
            inp_data[start + 1 : end + 1, :64]
            for start, end in zip(frame_start_indices, frame_end_indices)
        ]
    )
    inp_data = inp_data.reshape(-1, column_size, nreps, row_size)
    return inp_data


def _write_data_to_h5(path, data, attributes=None) -> None:
    """
    Writes data to a h5 file.
    Takes the full path of a dataset: /scratch/some_folder/data.h5/some_group/some_dataset
    Args:
        dataset_path: the path to the dataset including the path to the file
        data: the data to write
        exists_error: if True, raises an error if the dataset already exists, if false it does nothing
    """
    h5_file, dataset_path = utils.split_h5_path(path)
    with h5py.File(h5_file, "a") as f:
        path_parts = dataset_path.strip("/").split("/")
        groups = path_parts[:-1]
        dataset = path_parts[-1]
        current_group = f
        for group in groups:
            if group not in current_group:
                current_group = current_group.create_group(group)
            else:
                current_group = current_group[group]
        if dataset in current_group:
            raise ValueError(f"Dataset {dataset} already exists in {h5_file}")
        else:
            dataset = current_group.create_dataset(
                dataset, dtype=data.dtype, data=data, chunks=None
            )
            if attributes is not None:
                for key, value in attributes.items():
                    dataset.attrs[key] = value


def _read_data_from_h5(path: str) -> np.ndarray:
    h5_file, dataset_path = utils.split_h5_path(path)
    with h5py.File(h5_file, "r") as f:
        data = f[dataset_path][:]
    return data


def _get_datasets_from_h5(path: str):
    """
    Returns a list of datasets in a group.
    Args:
        group_path: path to the group
    Returns:
        datasets: list of datasets and its shapes in the group
    """
    h5_file, group_path = utils.split_h5_path(path)
    with h5py.File(h5_file, "r") as f:
        datasets = []
        group = f[group_path]
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                datasets.append([name, list(item.shape), dict(item.attrs)])
    return datasets


def _process_raw_data(
    h5_group,
    column_size,
    row_size,
    key_ints,
    ignore_first_nreps,
    ext_dark_frame_h5,
    nreps,
    offset,
    counts,
    bin,
    batch_index,
    process_index,
    nreps_eval,
) -> None:
    # read data from bin file, multiple processes can read from the same file
    # write the avg attribute to the dset to determine what to average later in the vds
    try:
        data = _read_data_from_bin(
            bin, column_size, row_size, key_ints, nreps, offset, counts
        )
        _write_data_to_h5(h5_group + "raw_data", data, {"avg":"False"})
        data = data[:, :, ignore_first_nreps:, :]
        #raw_set is multiplied with #frames to calculated the weighted average later
        raw_offset = np.mean(data, axis=0, keepdims=True)*data.shape[0]
        raw_data_mean = np.mean(data, axis=2)
        raw_data_median = np.median(data, axis=2)
        raw_data_std = np.std(data, axis=2)
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
    h5_group,
    h5_file_virtual,
    ignore_first_nreps,
    ext_dark_frame_dset,
    offset,
    batch_index,
    process_index,
    nreps_eval,
) -> None:
    # read data from bin file, multiple processes can read from the same file
    try:
        data = _read_data_from_h5(h5_group + "raw_data")
        data = data[:, :, ignore_first_nreps:, :]
        data = data.astype(np.float64)
        if ext_dark_frame_dset is not None:
            offset = _read_data_from_h5(ext_dark_frame_dset)
        else:
            offset = _read_data_from_h5(h5_file_virtual + "raw_offset_weighted_frames")
        data -= offset
        common_modes = np.median(data, axis=3, keepdims=True)
        data -= common_modes
        _write_data_to_h5(h5_group + "preproc_common_modes", common_modes, {"avg": "mean"})
        del offset, common_modes
        gc.collect()
        if data.shape[2] >50:
            shapiro = utils.shapiro(data, axis=2)
            _write_data_to_h5(h5_group + "preproc_shapiro", shapiro, {"avg": "mean"})
            del shapiro
        mean = np.mean(data, axis=2)
        std = np.std(data, axis=2)
        median = np.median(data, axis=2)
        # TODO: Slopes are slow when calculated like this, maybe use numba?
        x = np.arange(data.shape[2])
        slopes = np.apply_along_axis(lambda y: np.polyfit(x, y, 1)[0], axis=2, arr=data)
        _write_data_to_h5(h5_group + "preproc_mean_nreps", mean, {"avg": "mean"})
        _write_data_to_h5(h5_group + "preproc_median_nreps", median, {"avg": "mean"})
        _write_data_to_h5(h5_group + "preproc_std_nreps", std, {"avg": "mean"})
        _write_data_to_h5(h5_group + "preproc_slope_nreps", slopes, {"avg": "mean"})
        del mean, std, median, slopes
        gc.collect()
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
            slopes = np.apply_along_axis(
                lambda y: np.polyfit(x, y, 1)[0], axis=2, arr=data_slice
            )
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
    bin_files: List[str],
    output_folder: str,
    column_size: int = 64,
    row_size: int = 64,
    key_ints: int = 3,
    ignore_first_nreps: int = 3,
    offset: int = 8,
    available_cpu_cores: int = 4,
    available_ram_gb: int = 16,
    ext_dark_frame_h5: str = None,
    nreps_eval: List[List[int]] = [],
    attributes: dict = None,
) -> None:
    # check if folder, bin files exist and calculate nreps and make sure they are all the same
    if not os.path.exists(output_folder):
        raise Exception(f"Folder {output_folder} does not exist.")
    nreps_list = []
    for bin_file in bin_files:
        if not os.path.exists(bin_file):
            raise Exception(f"File {bin_file} does not exist")
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
        raise Exception(
            f"Not all bin files have the same number of nreps: {nreps_list}"
        )
    # check if external dark frame exists and has the right shape
    if ext_dark_frame_h5 is not None:
        ext_h5_file = ext_dark_frame_h5.split(".h5")[0] + ".h5"
        ext_group_path = ext_dark_frame_h5.split(".h5")[1]
        if not os.path.exists(ext_h5_file):
            raise Exception(f'File "{ext_h5_file}" does not exist')
        with h5py.File(ext_h5_file, "r") as f:
            try:
                shape = f[ext_group_path].shape
            except Exception as e:
                raise Exception(
                    f"Could not read shape of external dark frame {ext_dark_frame_h5}: {e}"
                )
            if shape[0] != column_size or shape[1] != row_size:
                raise Exception(
                    f"Shape of external dark frame {ext_dark_frame_h5} does"
                    "not match ({column_size}, {row_size}) of the bin files"
                )
    # leave one core for the main process
    # available_cpu_cores -= 1
    # create folders:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    bin_name = os.path.basename(bin_files[0]).split(".")[0]
    working_folder = os.path.join(output_folder, f"{timestamp}_{bin_name}")
    data_folder = os.path.join(working_folder, "data")
    os.mkdir(working_folder)
    os.mkdir(data_folder)
    # create a h5 file for every process in the data_folder
    h5_file_process = [
        os.path.join(data_folder, f"data_{i}.h5") for i in range(available_cpu_cores)
    ]

    for file in h5_file_process:
        with h5py.File(file, "w") as f:
            f.attrs["info"] = (
                # TODO: add info about bin files to this.
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
    _logger.info("These .bin files will be processed:")
    for bin in bin_files:
        _logger.info(f"{bin} of size {(os.path.getsize(bin) / (1024**3)):.2f} GB")
        _logger.info(
            f"The file will be split into {len(workload_dict[bin])} batches to fit into memory."
        )
    _logger.info(
        "Note, that the provided bin files will be treated as being from the same measurement."
    )
    _logger.info(
        "If you wish to process multiple measurements, please provide them separately."
    )
    _logger.info("Start processing Raw Data.")
    for bin in workload_dict.keys():
        _logger.info(f"Start processing {bin}")
        for batch_index, batch in enumerate(workload_dict[bin]):
            processes = []
            for process_index, [offset, counts, h5_group] in enumerate(batch):
                p = multiprocessing.Process(
                    target=_process_raw_data,
                    args=(
                        h5_group,
                        column_size,
                        row_size,
                        key_ints,
                        ignore_first_nreps,
                        ext_dark_frame_h5,
                        nreps,
                        offset,
                        counts,
                        bin,
                        batch_index,
                        process_index,
                        nreps_eval,
                    ),
                )
                processes.append(p)
                p.start()
            _logger.info(
                f"batch {batch_index+1}/{len(workload_dict[bin])} started, "
                f"{available_cpu_cores} processes are running."
            )
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
    for bin in workload_dict.keys():
        _logger.info(f"Start processing {bin}")
        for batch_index, batch in enumerate(workload_dict[bin]):
            processes = []
            for process_index, [offset, counts, h5_group] in enumerate(batch):
                p = multiprocessing.Process(
                    target=_preprocess,
                    args=(
                        h5_group,
                        h5_file_virtual,
                        ignore_first_nreps,
                        ext_dark_frame_h5,
                        offset,
                        batch_index,
                        process_index,
                        nreps_eval,
                    ),
                )
                processes.append(p)
                p.start()
            _logger.info(
                f"batch {batch_index+1}/{len(workload_dict[bin])} started, "
                f"{available_cpu_cores} processes are running."
            )
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
    _logger.info(f"Final Dataset is stored in {h5_file_virtual}")
    _logger.info(f"Data is stored in {data_folder}")
    _logger.info(
        "DO NOT move the data folder, if you do the virtual dataset will not work anymore."
    )
    _logger.info("You can move the .h5 file to any location.\n")
