from functools import partial
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
import tables

from . import utils


def label_frame(
    data: np.ndarray,
    primary_threshold: float,
    secondary_threshold: float,
    noise_map: np.ndarray,
    structure: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Label connected regions in the data based on primary and secondary thresholds.

    Args:
        data: 2D array of pixel values for a single frame
        primary_threshold: Primary threshold value
        secondary_threshold: Secondary threshold value
        noise_map: 2D array of noise values corresponding to the data
        structure: Structure array for morphological operations
    """

    primary_mask = data > primary_threshold * noise_map
    secondary_mask = data > secondary_threshold * noise_map

    # Label primary regions
    primary_labels, num_primary = ndimage.label(primary_mask, structure=structure)  # type: ignore

    # For each primary region, grow into secondary threshold areas
    final_labels = np.zeros_like(data, dtype=np.uint16)

    for i in range(1, num_primary + 1):
        # Create seed from primary region
        seed = primary_labels == i

        # Dilate iteratively while staying within secondary threshold
        current_region = seed.copy()

        while True:
            # Dilate one step
            dilated = ndimage.binary_dilation(current_region, structure=structure)
            # Keep only pixels above secondary threshold
            new_region = np.logical_and(dilated, secondary_mask)

            # Stop if no new pixels added
            if np.array_equal(new_region, current_region):
                break
            current_region = new_region

        final_labels[current_region] = i
    num_features = int(np.max(final_labels)) if final_labels.max() > 0 else 0

    return final_labels, num_features


def write_event_data_to_h5(event_data: list, h5_filename: str, path: str, table_name: str) -> None:
    """
    Write event data to an HDF5 file using PyTables.

    Args:
        event_data: List of event dictionaries from create_event_data()
        h5_filename: Path to the HDF5 file to create/write
        table_name: Name of the table in the HDF5 file (default: "events")
    """

    class EventRecord(tables.IsDescription):
        signal_id = tables.UInt32Col(dflt=0)  # type: ignore
        event_id = tables.UInt32Col(dflt=0)  # type: ignore
        no_signals = tables.UInt16Col(dflt=0)  # type: ignore
        frame = tables.UInt32Col(dflt=0)  # type: ignore
        row = tables.UInt8Col(dflt=0)  # type: ignore
        col = tables.UInt8Col(dflt=0)  # type: ignore
        value = tables.Float64Col(dflt=0.0)  # type: ignore
        is_primary = tables.BoolCol(dflt=False)  # type: ignore
        is_secondary = tables.BoolCol(dflt=False)  # type: ignore

    # Create and write to HDF5 file
    with tables.open_file(h5_filename, mode="a") as h5file:
        # Create the group path if it doesn't exist
        if path != "/" and not path in h5file:
            # Handle nested paths (e.g., "/analysis/events")
            path_parts = [p for p in path.split("/") if p]  # Remove empty strings
            current_path = "/"

            for part in path_parts:
                new_path = current_path + part if current_path == "/" else current_path + "/" + part
                if new_path not in h5file:
                    h5file.create_group(current_path, part)
                current_path = new_path
        # Create the table
        table = h5file.create_table(path, table_name, EventRecord, "Pixel events with thresholds")

        # Get a reference to table row for writing
        event_row = table.row

        # Write each event to the table
        for event in event_data:
            # Fill the row data
            event_row["signal_id"] = event["signal_id"]
            event_row["event_id"] = event["event_id"]
            event_row["no_signals"] = event["no_signals"]
            event_row["frame"] = event["frame"]
            event_row["row"] = event["row"]
            event_row["col"] = event["col"]
            event_row["value"] = event["value"]
            event_row["is_primary"] = event["is_primary"]
            event_row["is_secondary"] = event["is_secondary"]

            # Add the row to the table
            event_row.append()

        # Flush data to disk
        table.flush()

        # Add some metadata
        table.attrs.total_events = len(event_data)
        table.attrs.description = "Event data from two-threshold analysis"

        print(f"Successfully wrote {len(event_data)} events to {h5_filename}")


def process_frame_batch(frame_indices, data, primary_threshold, secondary_threshold, noise_map, structure):
    """
    Process a batch of frames and return event data for those frames.

    Args:
        frame_indices: List of frame indices to process
        data: Full data array
        primary_threshold: Primary threshold value
        secondary_threshold: Secondary threshold value
        noise_map: Noise map array
        structure: Structure array for morphological operations

    Returns:
        List of event dictionaries for the processed frames
    """
    batch_event_data = []

    for frame_idx in frame_indices:
        # get all labels for the frame
        labels, num_features = label_frame(
            data[frame_idx, :], primary_threshold, secondary_threshold, noise_map, structure
        )

        # Store frame events with frame info for later sorting
        frame_events = []
        signal_counter = 0
        event_counter = 0
        # loop through features (0=no feature)
        for feature in range(1, num_features + 1):
            # Find all labeled pixels with that feature
            labeled_pixels = np.where(labels == feature)
            no_signals = len(labeled_pixels[0])

            for i in range(len(labeled_pixels[0])):
                row_idx = labeled_pixels[0][i]
                col_idx = labeled_pixels[1][i]
                pixel_value = data[frame_idx, row_idx, col_idx]

                # Determine threshold levels
                primary_check = pixel_value > primary_threshold * noise_map[row_idx, col_idx]
                secondary_check = pixel_value > secondary_threshold * noise_map[row_idx, col_idx]

                frame_events.append(
                    {
                        "signal_id": signal_counter,
                        "event_id": event_counter,
                        "no_signals": no_signals,
                        "frame": frame_idx,
                        "row": row_idx,
                        "col": col_idx,
                        "value": pixel_value,
                        "is_primary": primary_check,
                        "is_secondary": np.bool_(secondary_check and not primary_check),
                    }
                )
                signal_counter += 1
            event_counter += 1
        batch_event_data.extend(frame_events)

    return batch_event_data


def divide_frames_evenly(total_frames: int, num_processes: int) -> list:
    """Divide frames evenly among processes."""
    quotient, remainder = divmod(total_frames, num_processes)
    result = [quotient] * num_processes
    for i in range(remainder):
        result[i] += 1
    return result


def create_event_data(
    data: np.ndarray,
    primary_threshold: float,
    secondary_threshold: float,
    noise_map: np.ndarray,
    structure: np.ndarray,
    available_cpus: int = 0,
) -> list:
    """
    Create event data from input arrays using multiprocessing Pool.

    Args:
        data: Input data array with shape (nframes, height, width)
        primary_threshold: Primary threshold value
        secondary_threshold: Secondary threshold value
        noise_map: Noise map array
        structure: Structure array for morphological operations
        available_cpus: (optional) Number of CPU cores to use for processing

    Returns:
        List of event dictionaries
    """
    if available_cpus == 0:
        available_cpus = utils.get_cpu_count()
        print(f"CPUS used: {available_cpus}")

    total_frames = data.shape[0]
    print(f"Processing {total_frames} frames using {available_cpus} CPUs")

    # Divide frames among available CPUs
    frames_per_cpu = divide_frames_evenly(total_frames, available_cpus)

    # Create frame index batches for each process
    frame_batches = []
    start_idx = 0
    for cpu_frames in frames_per_cpu:
        end_idx = start_idx + cpu_frames
        if start_idx < total_frames:  # Only create batch if there are frames to process
            frame_batches.append(list(range(start_idx, min(end_idx, total_frames))))
        start_idx = end_idx

    print(f"Created {len(frame_batches)} batches")

    # Create partial function with fixed parameters
    # TODO: dont copy all the data!
    process_func = partial(
        process_frame_batch,
        data=data,
        primary_threshold=primary_threshold,
        secondary_threshold=secondary_threshold,
        noise_map=noise_map,
        structure=structure,
    )

    # Process batches in parallel using Pool.map
    with Pool(processes=available_cpus) as pool:
        print("Starting parallel processing...")
        batch_results = pool.map(process_func, frame_batches)
        print("Parallel processing complete.")

    all_events = []
    for batch in batch_results:
        all_events.extend(batch)

    print(f"Processed {len(all_events)} signals")
    return all_events
