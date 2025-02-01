import h5py
import numpy as np
import json
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import gc
from multiprocessing import Manager
import time

from . import logger
from . import utils
from . import file_io as io

_logger = logger.Logger(__name__, "info").get_logger()


def create_data_file_from_bins_v2(
    bin_files: List[str],
    output_folder: str,
    output_filename: str,
    nreps: int,
    column_size: int = 64,
    row_size: int = 64,
    key_ints: int = 3,
    offset: int = 8,
    attributes: dict = None,
    dark_frame_offset: str = None,
    compression: str = None,
    available_cpu_cores: int = 32,
    available_ram_gb: int = 12,
) -> None:
    """
    All sizes are in bytes, unsless otherwise specified.
    """
    output_file = os.path.join(output_folder, output_filename)
    # check if h5 file already exists
    if os.path.exists(output_file):
        _logger.error(f"File {output_file} already exists. Please delete")
        raise Exception(f"File {output_file} already exists. Please delete")
    # check if bin files exist, nreps are consistent and add up the total size
    bin_size_list = []
    for bin_file in bin_files:
        if not os.path.exists(bin_file):
            _logger.error(f"File {bin_file} does not exist")
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
            # add up the total size
            bin_size_list.append(os.path.getsize(bin_file))
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
        # we use 30% of the available RAM to be safe
        available_ram = int((available_ram_gb * 1024 * 1024 * 1024) * 0.3)
        available_ram_per_process = int(available_ram / available_cpu_cores)
        # calculate how many rows can be read per process, a row has 67 uint16 values 2 bytes each
        rows_read_per_process = int(
            available_ram_per_process / ((row_size + key_ints) * 2)
        )
        process_infos = {}
        """
        Structure of process_infos:
        {
            "bin1": value[round_index][process_index] = (offset (in bytes), counts (in uint16))
            "bin2": value[round_index][process_index] = (offset (in bytes), counts (in uint16))
            ...
        }
        """
        for bin in bin_files:
            bin_list = []
            # bin_size is in unit bytes
            bin_size = os.path.getsize(bin)
            # check how often all subprocesses must be called to read the whole file
            rounds = int(bin_size / available_ram) + 1
            # the offset is named that way because of the kwarg of np.fromfile
            current_offset = offset
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
            process_infos[bin] = bin_list
        print(process_infos)
        manager = Manager()
        shared_dict = manager.dict()
        """
        Structure of shared_dict:
        {
            "bin1": value[round_index][process_index] = read frames in subprocess
            "bin2": calue[round_index][process_index] = read frames in subprocess
            ...
        }
        """
        for bin in process_infos.keys():
            shared_dict[bin] = manager.list(
                [manager.list() for _ in range(len(process_infos[bin]))]
            )

        for key, value in shared_dict.items():
            print(f"{key}: {[list(inner_list) for inner_list in value]}")

        print("start process spawning")
        for bin in process_infos.keys():
            print(f"start processing {bin}")
            for round_id, chunk in enumerate(process_infos[bin]):
                print(f"start round {round_id}")
                with ProcessPoolExecutor(max_workers=available_cpu_cores) as executor:
                    futures = []
                    for process_id, offset_tuple in enumerate(chunk):
                        futures.append(
                            executor.submit(
                                read_and_process_data_from_bin,
                                bin,
                                round_id,
                                process_id,
                                column_size,
                                row_size,
                                key_ints,
                                nreps,
                                offset_tuple[0],
                                offset_tuple[1],
                                shared_dict,
                                available_cpu_cores,
                                output_file,
                            )
                        )
                    gc.collect()
                    for future in as_completed(futures):
                        future.result()
                    gc.collect()

    for key, value in shared_dict.items():
        print(f"{key}: {[list(inner_list) for inner_list in value]}")


def read_and_process_data_from_bin(
    bin_file: str,
    round_id: int,
    process_id: int,
    column_size: int,
    row_size: int,
    key_ints: int,
    nreps: int,
    offset: int,
    chunk_size: int,
    shared_dict: dict,
    available_cpu_cores: int,
    output_file: str,
) -> Tuple[np.ndarray, int]:
    """
    Reads data from a .bin file in chunks and returns a numpy array with the data and the new offset.
    The bins usually have a header of 8 bytes, followed by the data. So an initial offset of 8 is needed.
    After that the new offset is the position in the file where the next chunk should start. The parameter
    frames_to_read should be set to a value that fits into the available RAM.
    The function returns -1 as offset when end of file is reached.
    offset must be in bytes, chunk_size must be in bytes as well.

    Returns:
        data, offset: Tuple[np.ndarray, int]
    """
    raw_row_size = row_size + key_ints
    rows_per_frame = column_size * nreps
    # count parameter needs to be in units of uint16 (uint16 = 2 bytes)
    inp_data = np.fromfile(
        bin_file, dtype="uint16", count=int(chunk_size / 2), offset=offset
    )
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
        _logger.warning("No valid frames found in chunk!")
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
    final_frame_count = inp_data.shape[0]
    shared_dict[bin_file][round_id].append((process_id, final_frame_count))
    # wait until all processes are done reading
    processed_finished = False
    while not processed_finished:
        processed_finished = True
        if len(shared_dict[bin_file][round_id]) != available_cpu_cores:
            processed_finished = False
        time.sleep(1)
    # now lets find the starting index for writing to .h5 file
    index = 0
    last_bin = False
    last_round = False
    for bin_name, value in shared_dict.items():
        if last_bin == True:
            break
        for i in range(round_id + 1):
            if i == round_id:
                last_round = True
            for process in value[i]:
                current_id = process[0]
                frames_found = process[1]
                if last_round:
                    if current_id < process_id:
                        index += frames_found
                else:
                    index += frames_found
    if bin_name == bin_file:
        last_bin = True
    print(f"Round {round_id}, Process {process_id}, Started writing from index {index}")
    # TODO: this takes waaay too long
    io.add_array_to_file(output_file, "data", inp_data, index)
    print(
        f"Round {round_id}, Process {process_id}, Finished writing from index {index}"
    )
