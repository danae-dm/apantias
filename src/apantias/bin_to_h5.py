import h5py
import numpy as np
import multiprocessing
import time
from typing import List, Tuple, Optional
import os


def _get_workload_dict(
    bin_files, available_ram_gb, available_cpu_cores, row_size, key_ints, initial_offset
) -> dict:
    """
    Returns a dictionary with the workload for each process. It looks at the size if the bin files and
    assigns bits of the file to each process. Each process has available_ram/available_cpu_cores ram to work with.
    The workload is a list of #available_cpu_cores tuples with the offset and the number of uint16 values to read.
    If the file is larger than available_ram, the file is loaded in multiple "rounds".
    {
        'bin_file1': [
            [(offset1, counts1), (offset2, counts2), ...],  # round 1 of available_cpu_cores processes
            [(offset1, counts1), (offset2, counts2), ...],  # round 2 of available_cpu_cores processes, if needed
            ...
        ],
        'bin_file2': [                                      # same for the next bin file
            [(offset1, counts1), (offset2, counts2), ...],
            [(offset1, counts1), (offset2, counts2), ...],
            ...
        ],
        ...
        Args:
            bin_files: list of absolute paths to the bin files
            available_ram: available ram per process in bytes
            available_cpu_cores: number of available cpu cores
            rows_read_per_process: number of rows to read per process
            row_size: size of a row in bytes
            key_ints: number of key integers
            offset: offset in bytes to start reading
        Returns:
            workload_dict: dictionary with the workload for each process
    """
    # 30% of available ram, this number can be tweaked later to improve performance
    available_ram = int((available_ram_gb * 1024 * 1024 * 1024) * 0.3)
    available_ram_per_process = int(available_ram / available_cpu_cores)
    rows_read_per_process = int(available_ram_per_process / ((row_size + key_ints) * 2))
    workload_dict = {}
    for bin in bin_files:
        bin_list = []
        # bin_size is in unit bytes
        bin_size = os.path.getsize(bin)
        # check how often all subprocesses must be called to read the whole file
        rounds = int(bin_size / available_ram) + 1
        # the offset is named that way because of the kwarg of np.fromfile
        current_offset = initial_offset
        for i in range(rounds):
            bin_list.append([])
            bytes_left = bin_size - current_offset
            if bytes_left > available_ram:
                for n in range(available_cpu_cores):
                    # counts is the number of uint16 values to read
                    counts = rows_read_per_process * (row_size + key_ints)
                    bin_list[i].append((current_offset, counts))
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
                    bin_list[i].append((current_offset, counts))
                    # set the new offset in bytes
                    current_offset += counts * 2
        workload_dict[bin] = bin_list
    return workload_dict


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
    inp_data = inp_data.reshape(-1, raw_row_size)
    # find all the framekeys
    frame_keys = np.where(inp_data[:, column_size] == 65535)
    # stack them and calculate difference to find incomplete frames
    frames = np.stack((frame_keys[0][:-1], frame_keys[0][1:]))
    diff = np.diff(frames, axis=0)
    valid_frames_position = np.nonzero(diff == rows_per_frame)[1]
    if len(valid_frames_position) == 0:
        print("No valid frames found in chunk!")
        return None
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


def _write_data_to_h5(lock, h5_file, dataset_path, data) -> None:
    """ "
    Writes data to a h5 file in a thread safe manner.
    If the dataset does not exist, it is created with unlimited
    maxshape along axis=0.
    It appends the data to the dataset along axis=0.
    Args:
        lock: a multiprocessing lock
        h5_file: the path to the h5 file
        dataset_path: the path to the dataset in the h5 file
        data: the data to write
    """
    with lock:
        with h5py.File(h5_file, "a", libver="latest") as f:
            f.swmr_mode = True
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
                dataset = current_group[dataset]
            else:
                dataset = current_group.create_dataset(
                    dataset,
                    dtype=data.dtype,
                    shape=(0, *data.shape[1:]),
                    maxshape=(None, *data.shape[1:]),
                    chunks=None,
                )
            dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
            dataset[-data.shape[0] :] = data
            f.flush()


def _run_subprocess(
    h5_file,
    column_size,
    row_size,
    key_ints,
    nreps,
    offset,
    counts,
    shared_dict,
    bin,
    round_index,
    process_index,
    lock,
) -> None:
    # read data from bin file, multiple processes can read from the same file
    data = _read_data_from_bin(
        bin, column_size, row_size, key_ints, nreps, offset, counts
    )
    common_modes = np.nanmedian(data, axis=3)

    # now, check if all processes with a smaller id have finished writing, if not, wait
    # this acts as a barrier
    writing_permitted = False
    while not writing_permitted:
        writing_permitted = True
        # check if all processes with a smaller id have finished writing
        for process in shared_dict[bin][round_index]:
            if process < process_index:
                writing_permitted = False
    # write data to h5 file, this is done by only one process at a time

    _write_data_to_h5(lock, h5_file, "data", data)
    _write_data_to_h5(lock, h5_file, "common_modes", common_modes)
    shared_dict[bin][round_index].remove(process_index)


def create_data_file_from_bins(
    bin_files: List[str],
    h5_file: str,
    nreps: int,
    column_size: int = 64,
    row_size: int = 64,
    key_ints: int = 3,
    offset: int = 8,
    available_cpu_cores: int = 4,
    available_ram_gb: int = 16,
    ext_dark_frame: str = None,
    attributes: dict = None,
) -> None:
    # check if bin files exist, nreps are consistent and add up the total size
    if os.path.exists(h5_file):
        print(f"File {h5_file} already exists. Please delete")
        raise Exception(f"File {h5_file} already exists. Please delete")
    for bin_file in bin_files:
        if not os.path.exists(bin_file):
            print(f"File {bin_file} does not exist")
            raise Exception(f"File {bin_file} does not exist")
        else:
            # check if nreps are correct
            raw_row_size = row_size + key_ints
            test_data = np.fromfile(
                bin_file,
                dtype="uint16",
                # load three times the given nreps and load 20 frames, times two because uint16 is 2 bytes
                count=column_size * raw_row_size * 3 * nreps * 10 * 2,
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
            if nreps != estimated_nreps:
                raise Exception(
                    f"Estimated nreps for {bin_file}: {estimated_nreps}, given nreps: {nreps}"
                )
            else:
                print(f"Estimated nreps for {bin_file}: {estimated_nreps}")
    workload_dict = _get_workload_dict(
        bin_files, available_ram_gb, available_cpu_cores, row_size, key_ints, offset
    )
    manager = multiprocessing.Manager()
    # initialize shared dict for synchronization
    shared_dict = manager.dict()
    for bin in workload_dict.keys():
        shared_dict[bin] = manager.list(
            [
                manager.list(range(available_cpu_cores))
                for _ in range(len(workload_dict[bin]))
            ]
        )
    for key, value in shared_dict.items():
        print(f"{key}: {[list(inner_list) for inner_list in value]}")
    # create new h5 file with swmr mode
    with h5py.File(h5_file, "w", libver="latest") as f:
        f.create_dataset(
            "data",
            dtype="uint16",
            shape=(0, 64, 200, 64),
            maxshape=(None, 64, 200, 64),
            chunks=None,
        )
        f.swmr_mode = True

    lock = multiprocessing.Lock()
    processes = []
    for bin in workload_dict.keys():
        for round_index, round in enumerate(workload_dict[bin]):
            for process_index, (offset, counts) in enumerate(round):
                p = multiprocessing.Process(
                    target=_run_subprocess,
                    args=(
                        h5_file,
                        column_size,
                        row_size,
                        key_ints,
                        nreps,
                        offset,
                        counts,
                        shared_dict,
                        bin,
                        round_index,
                        process_index,
                        lock,
                    ),
                )
                processes.append(p)
                p.start()
            print(f"Waiting for round {round_index} to finish")
            for p in processes:
                p.join()
            print(f"Round {round_index} finished")
