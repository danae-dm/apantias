import gc
import os
import psutil
from datetime import datetime

import numpy as np

from . import logger
from . import utils
from . import analysis as an
from . import params
from . import fitting as fit
from . import file_io as io


class RoanSteps:
    _logger = logger.Logger("nproan-RoanSteps", "info").get_logger()

    def __init__(self, prm_file: str, ram: int) -> None:
        self.ram_available = ram
        self.load(prm_file)

    def load(self, prm_file: str) -> None:
        # load parameter file
        self.params = params.Params(prm_file)
        self.params_dict = self.params.get_dict()

        # polarity is from the old code, im not quite sure why it is -1
        self.polarity = -1

        # common parameters from params file
        self.results_dir = self.params_dict["common_results_dir"]
        self.bad_pixels = self.params_dict["common_bad_pixels"]

        # offnoi parameters from params file
        self.offnoi_data_file = self.params_dict["offnoi_data_file"]
        self.offnoi_nframes_eval = self.params_dict["offnoi_nframes_eval"]
        self.offnoi_nreps_eval = self.params_dict["offnoi_nreps_eval"]
        self.offnoi_comm_mode = self.params_dict["offnoi_comm_mode"]
        self.offnoi_thres_mips = self.params_dict["offnoi_thres_mips"]
        self.offnoi_thres_bad_frames = self.params_dict["offnoi_thres_bad_frames"]
        self.offnoi_thres_bad_slopes = self.params_dict["offnoi_thres_bad_slopes"]

        # filter parameters from params file
        self.filter_data_file = self.params_dict["filter_data_file"]
        self.filter_nframes_eval = self.params_dict["filter_nframes_eval"]
        self.filter_nreps_eval = self.params_dict["filter_nreps_eval"]
        self.filter_comm_mode = self.params_dict["filter_comm_mode"]
        self.filter_thres_mips = self.params_dict["filter_thres_mips"]
        self.filter_thres_event_prim = self.params_dict["filter_thres_event_prim"]
        self.filter_thres_event_sec = self.params_dict["filter_thres_event_sec"]
        self.filter_use_fitted_offset = self.params_dict["filter_use_fitted_offset"]
        self.filter_thres_bad_frames = self.params_dict["filter_thres_bad_frames"]
        self.filter_thres_bad_slopes = self.params_dict["filter_thres_bad_slopes"]

        # parameters from data_h5 files
        total_frames_offnoi, column_size_offnoi, row_size_offnoi, nreps_offnoi = (
            io.get_params_from_data_file(self.offnoi_data_file)
        )
        total_frames_filter, column_size_filter, row_size_filter, nreps_filter = (
            io.get_params_from_data_file(self.filter_data_file)
        )
        # check if sensor size is equal
        if (
            column_size_offnoi != column_size_filter
            or row_size_offnoi != row_size_filter
        ):
            raise ValueError(
                "Column size or row size of offnoi and filter data files are not equal."
            )

        self.column_size = column_size_offnoi
        self.row_size = row_size_offnoi
        # set total number of frames and nreps from the data file
        self.offnoi_total_nreps = nreps_offnoi
        self.offnoi_total_frames = total_frames_offnoi
        self.filter_total_nreps = nreps_filter
        self.filter_total_frames = total_frames_filter

        # nreps_eval and nframes_eval is [start,stop,step], if stop is -1 it goes to the end
        if self.offnoi_nframes_eval[1] == -1:
            self.offnoi_nframes_eval[1] = self.offnoi_total_frames
        if self.offnoi_nreps_eval[1] == -1:
            self.offnoi_nreps_eval[1] = self.offnoi_total_nreps
        if self.filter_nframes_eval[1] == -1:
            self.filter_nframes_eval[1] = self.filter_total_frames
        if self.filter_nreps_eval[1] == -1:
            self.filter_nreps_eval[1] = self.filter_total_nreps

        # create slices, these are used to get the data from the data files
        self.offnoi_nreps_slice = slice(*self.offnoi_nreps_eval)
        self.offnoi_nframes_slice = slice(*self.offnoi_nframes_eval)
        self.filter_nreps_slice = slice(*self.filter_nreps_eval)
        self.filter_nframes_slice = slice(*self.filter_nframes_eval)

        # set variables to number of nreps_eval and nframes_eval to be evaluated (int)
        self.offnoi_nreps_eval = int(
            (self.offnoi_nreps_eval[1] - self.offnoi_nreps_eval[0])
            / self.offnoi_nreps_eval[2]
        )
        self.offnoi_nframes_eval = int(
            (self.offnoi_nframes_eval[1] - self.offnoi_nframes_eval[0])
            / self.offnoi_nframes_eval[2]
        )
        self.filter_nreps_eval = int(
            (self.filter_nreps_eval[1] - self.filter_nreps_eval[0])
            / self.filter_nreps_eval[2]
        )
        self.filter_nframes_eval = int(
            (self.filter_nframes_eval[1] - self.filter_nframes_eval[0])
            / self.filter_nframes_eval[2]
        )

        # check, if offnoi_nreps_eval is greater or equal than filter_nreps_eval
        # this is necessary, because the filter step needs the offset_raw from the offnoi step
        if self.offnoi_nreps_eval < self.filter_nreps_eval:
            raise ValueError(
                "offnoi_nreps_eval must be greater or equal than filter_nreps_eval"
            )

        # create analysis h5 file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        bin_filename = os.path.basename(self.offnoi_data_file)[:-3]
        self.analysis_file_name = f"{timestamp}_{bin_filename}.h5"
        self.analysis_file = os.path.join(self.results_dir, self.analysis_file_name)
        io.create_analysis_file(
            self.results_dir,
            self.analysis_file_name,
            self.offnoi_data_file,
            self.filter_data_file,
            self.params_dict,
        )
        self._logger.info(
            f"Created analysis h5 file: {self.results_dir}/{self.analysis_file_name}"
        )
        self._logger.info(f"Parameters loaded:")
        self.params.print_contents()

    def calc_offnoi_step(self) -> None:

        estimated_ram_usage = (
            utils.get_ram_usage_in_gb(
                self.offnoi_nframes_eval,
                self.column_size,
                self.offnoi_nreps_eval,
                self.row_size,
            )
            * 2.5  # this is estimated, better safe than sorry
        )
        self._logger.info(f"RAM available: {self.ram_available:.1f} GB")
        self._logger.info(f"Estimated RAM usage: {estimated_ram_usage:.1f} GB")
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f"Steps needed: {steps_needed}")

        # (planned) frames per step, so that ram usage is below the available ram
        frames_per_step = int(self.offnoi_nframes_eval / steps_needed)

        # total processed frames over all steps
        total_frames_processed = 0

        for step in range(steps_needed):
            self._logger.info(f"Start processing step {step+1} of {steps_needed}")
            current_frame_slice = slice(
                total_frames_processed,
                total_frames_processed + frames_per_step,
            )
            slices = [
                current_frame_slice,
                slice(None),
                self.offnoi_nreps_slice,
                slice(None),
            ]
            data = (
                io.get_data_from_file(self.offnoi_data_file, "data", slices)
                * self.polarity
            )
            self._logger.info(f"Data loaded: {data.shape}")
            # set values of all frames and nreps of bad pixels to nan
            if self.bad_pixels:
                data = an.set_bad_pixellist_to_nan(data, self.bad_pixels)
            # delete bad frames from data
            if self.offnoi_thres_bad_frames != 0 or self.offnoi_thres_mips != 0:
                data = an.exclude_mips_and_bad_frames(
                    data, self.offnoi_thres_mips, self.offnoi_thres_bad_frames
                )
                self._logger.debug(f"Shape of data: {data.shape}")
            # Calculate offset_raw on the raw data and update file
            avg_over_frames = utils.get_avg_over_frames(data)
            io.add_array(self.analysis_file, "offnoi/offset_raw", avg_over_frames)
            # offset the data and correct for common mode if necessary
            data -= avg_over_frames[np.newaxis, :, :, :]
            if self.offnoi_comm_mode is True:
                an.correct_common_mode(data)
            # calculate rndr signals and update file
            avg_over_nreps = utils.get_avg_over_nreps(data)
            io.add_array(self.analysis_file, "offnoi/rndr_signals", avg_over_nreps)
            # calculate bad slopes and update file
            if self.offnoi_thres_bad_slopes != 0:
                slopes = an.get_slopes(data)
                io.add_array(self.analysis_file, "offnoi/slopes", slopes)
            total_frames_processed += frames_per_step
            self._logger.info(f"Finished step {step+1} of {steps_needed} total Steps")

        # TODO: paralellize this
        slopes = io.get_data_from_file(self.analysis_file, "offnoi/slopes")
        bad_slopes_pos = np.full(slopes.shape, False, dtype=bool)
        fit_params_list = []

        for row in range(slopes.shape[1]):
            for col in range(slopes.shape[2]):
                slopes_pixelwise = slopes[:, row, col]
                fit_pixelwise = fit.fit_gauss_to_hist(slopes_pixelwise.flatten())
                lower_bound = fit_pixelwise[1] - self.offnoi_thres_bad_slopes * np.abs(
                    fit_pixelwise[2]
                )
                upper_bound = fit_pixelwise[1] + self.offnoi_thres_bad_slopes * np.abs(
                    fit_pixelwise[2]
                )
                bad_slopes_mask = (slopes_pixelwise < lower_bound) | (
                    slopes_pixelwise > upper_bound
                )
                frame = np.where(bad_slopes_mask)[0]
                bad_slopes_pos[frame, row, col] = True
                fit_params_list.append(fit_pixelwise)
        bad_slopes_fit = np.vstack(fit_params_list)

        io.add_array(self.analysis_file, "offnoi/bad_slopes_pos", bad_slopes_pos)
        io.add_array(self.analysis_file, "offnoi/bad_slopes_fit", bad_slopes_fit)

        avg_over_nreps = io.get_data_from_file(
            self.analysis_file, "offnoi/rndr_signals"
        )
        avg_over_nreps_slopes_removed = avg_over_nreps.copy()
        avg_over_nreps_slopes_removed[bad_slopes_pos] = np.nan
        io.add_array(
            self.analysis_file,
            "offnoi/rndr_signals_slopes_removed",
            avg_over_nreps_slopes_removed,
        )

        self._logger.info("Fitting pixelwise for offset and noise")
        fitted = fit.get_fit_gauss(avg_over_nreps)
        io.add_array(self.analysis_file, "offnoi/fit", fitted)
        fitted = fit.get_fit_gauss(avg_over_nreps_slopes_removed)
        io.add_array(self.analysis_file, "offnoi/fit_slopes_removed", fitted)

    # TODO: continue here
    def calc_filter_step(self) -> None:
        estimated_ram_usage = (
            utils.get_ram_usage_in_gb(
                self.filter_nframes_eval,
                self.column_size,
                self.filter_nreps_eval,
                self.row_size,
            )
            * 2.5  # this is estimated, better safe than sorry
        )
        self._logger.info(f"RAM available: {self.ram_available:.1f} GB")
        self._logger.info(f"Estimated RAM usage: {estimated_ram_usage:.1f} GB")
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f"Steps needed: {steps_needed}")

        # (planned) frames per step, so that ram usage is below the available ram
        frames_per_step = int(self.filter_nframes_eval / steps_needed)

        # total processed frames over all steps
        total_frames_processed = 0

        for step in range(steps_needed):
            self._logger.info(f"Start processing step {step+1} of {steps_needed}")
            current_frame_slice = slice(
                total_frames_processed,
                total_frames_processed + frames_per_step,
            )
            slices = [
                current_frame_slice,
                slice(None),
                self.filter_nreps_slice,
                slice(None),
            ]
            data = (
                io.get_data_from_file(self.filter_data_file, "data", slices)
                * self.polarity
            )
            self._logger.info(f"Data loaded: {data.shape}")
            # set values of all frames and nreps of bad pixels to nan
            if self.bad_pixels:
                data = an.set_bad_pixellist_to_nan(data, self.bad_pixels)
            # delete bad frames from data
            if self.filter_thres_bad_frames != 0 or self.filter_thres_mips != 0:
                data = an.exclude_mips_and_bad_frames(
                    data, self.filter_thres_mips, self.filter_thres_bad_frames
                )
            self._logger.debug(f"Shape of data: {data.shape}")

            self.final_frames_per_step = data.shape[0]
            # Get offset_raw from offnoi step
            avg_over_frames_offnoi = io.get_data_from_file(
                self.analysis_file, "offnoi/offset_raw"
            )
            # offset the data and correct for common mode if necessary
            data -= avg_over_frames_offnoi[np.newaxis, :, : self.filter_nreps_eval, :]
            if self.filter_comm_mode is True:
                an.correct_common_mode(data)
            # calculate rndr signals and update file
            avg_over_nreps = utils.get_avg_over_nreps(data)
            # subtract fitted offset from data
            fitted_offset = io.get_data_from_file(self.analysis_file, "offnoi/fit")[1]
            avg_over_nreps -= fitted_offset
            io.add_array(self.analysis_file, "filter/rndr_signals", avg_over_nreps)
            # calculate bad slopes and update file
            if self.filter_thres_bad_slopes != 0:
                slopes = an.get_slopes(data)
                io.add_array(self.analysis_file, "filter/slopes", slopes)
            self._logger.info(f"Finished step {step+1} of {steps_needed} total Steps")
            total_frames_processed += frames_per_step

        slopes = io.get_data_from_file(self.analysis_file, "filter/slopes")
        bad_slopes_pos = np.full(slopes.shape, False, dtype=bool)
        fit_params_list = []

        for row in range(slopes.shape[1]):
            for col in range(slopes.shape[2]):
                slopes_pixelwise = slopes[:, row, col]
                fit_pixelwise = fit.fit_gauss_to_hist(slopes_pixelwise.flatten())
                lower_bound = fit_pixelwise[1] - self.offnoi_thres_bad_slopes * np.abs(
                    fit_pixelwise[2]
                )
                upper_bound = fit_pixelwise[1] + self.offnoi_thres_bad_slopes * np.abs(
                    fit_pixelwise[2]
                )
                bad_slopes_mask = (slopes_pixelwise < lower_bound) | (
                    slopes_pixelwise > upper_bound
                )
                frame = np.where(bad_slopes_mask)[0]
                bad_slopes_pos[frame, row, col] = True
                fit_params_list.append(fit_pixelwise)
        bad_slopes_fit = np.vstack(fit_params_list)
        io.add_array(self.analysis_file, "filter/bad_slopes_pos", bad_slopes_pos)
        io.add_array(self.analysis_file, "filter/bad_slopes_fit", bad_slopes_fit)

        avg_over_nreps_final = io.get_data_from_file(
            self.analysis_file, "filter/rndr_signals"
        )
        avg_over_nreps_slopes_removed = avg_over_nreps_final.copy()
        avg_over_nreps_slopes_removed[bad_slopes_pos] = np.nan
        io.add_array(
            self.analysis_file,
            "filter/rndr_signals_slopes_removed",
            avg_over_nreps_slopes_removed,
        )
