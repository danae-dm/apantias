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

        # get parameters from data_h5 file
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

        # create slices for retrieval of data from the data file
        # loading from h5 doesnt work with numpy sling notation, so we have to create slices
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
        self._logger.info(f"---------Start offnoi step---------")
        self._logger.info(f"RAM available: {self.ram_available:.1f} GB")
        self._logger.info(f"Estimated RAM usage: {estimated_ram_usage:.1f} GB")
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f"Steps needed: {steps_needed}")

        # (planned) frames per step, so that ram usage is below the available ram
        frames_per_step = int(self.offnoi_nframes_eval / steps_needed)

        # total processed frames over all steps
        total_frames_processed = 0

        # process this in steps, so that the ram usage is below the available ram
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

            # delete bad frames from data
            if self.offnoi_thres_bad_frames != 0 or self.offnoi_thres_mips != 0:
                data = an.exclude_mips_and_bad_frames(
                    data, self.offnoi_thres_mips, self.offnoi_thres_bad_frames
                )

            if self.offnoi_comm_mode is True:
                an.correct_common_mode(data)
            # calculate rndr signals and update file
            avg_over_nreps = utils.get_avg_over_nreps(data)
            io.add_array(
                self.analysis_file,
                "offnoi/precal/rndr_signals_after_common",
                avg_over_nreps,
            )
            # calculate bad slopes and update file
            if self.offnoi_thres_bad_slopes != 0:
                slopes = an.get_slopes(data)
                io.add_array(self.analysis_file, "offnoi/slopes/all_frames", slopes)
                io.add_array(
                    self.analysis_file,
                    "offnoi/slopes/average",
                    np.nanmean(slopes, axis=0),
                )
            total_frames_processed += frames_per_step
            self._logger.info(f"Finished step {step+1} of {steps_needed} total Steps")

        # TODO: paralellize this
        # Calculate the bad slopes
        self._logger.info("Start calculating bad slopes")
        slopes = io.get_data_from_file(self.analysis_file, "offnoi/slopes/all_frames")
        bad_slopes_pos = np.full(slopes.shape, False, dtype=bool)
        bad_slopes_fit = np.zeros((6, self.column_size, self.row_size))

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
                bad_slopes_fit[:, row, col] = fit_pixelwise

        io.add_array(self.analysis_file, "offnoi/slopes/bad_slopes_pos", bad_slopes_pos)
        io.add_array(
            self.analysis_file,
            "offnoi/slopes/bad_slopes_count",
            np.sum(bad_slopes_pos, axis=0),
        )
        io.add_array(self.analysis_file, "offnoi/slopes/bad_slopes_fit", bad_slopes_fit)
        self._logger.info("Finished calculating bad slopes")

        self._logger.info("Start calculating offset by fitting pixel wise")
        # load avg_over_nreps from the loop
        avg_over_nreps = io.get_data_from_file(
            self.analysis_file, "offnoi/precal/rndr_signals_after_common"
        )
        # set bad slopes to nan, so they not interfere in future calculations
        avg_over_nreps[bad_slopes_pos] = np.nan
        io.add_array(
            self.analysis_file,
            "offnoi/precal/rndr_signals_after_common_slopes_removed",
            avg_over_nreps,
        )
        # fit a 2 peak gaussian to the data
        fitted = fit.get_fit_over_frames(avg_over_nreps, peaks=2)
        io.add_array(self.analysis_file, "offnoi/fit/amplitude1", fitted[0, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/mean1", fitted[1, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/sigma1", fitted[2, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/error_amplitude1", fitted[3, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/error_mean1", fitted[4, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/error_sigma1", fitted[5, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/amplitude2", fitted[6, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/mean2", fitted[7, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/sigma2", fitted[8, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/error_amplitude2", fitted[9, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/error_mean2", fitted[10, :, :])
        io.add_array(self.analysis_file, "offnoi/fit/error_sigma2", fitted[11, :, :])
        # use the fitted mean of the first peak as offset
        avg_over_nreps -= fitted[1, :, :]

        io.add_array(
            self.analysis_file,
            "offnoi/rndr_signals/all_frames",
            avg_over_nreps,
        )
        io.add_array(
            self.analysis_file,
            "offnoi/rndr_signals/average",
            np.nanmean(avg_over_nreps, axis=0),
        )
        self._logger.info("Finished calculating offset by fitting pixel wise")
        self._logger.info("Finished offnoi step")

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
        self._logger.info(f"---------Start filter step---------")
        self._logger.info(f"RAM available: {self.ram_available:.1f} GB")
        self._logger.info(f"Estimated RAM usage: {estimated_ram_usage:.1f} GB")
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f"Steps needed: {steps_needed}")

        # (planned) frames per step, so that ram usage is below the available ram
        frames_per_step = int(self.filter_nframes_eval / steps_needed)

        # total processed frames over all steps
        total_frames_processed = 0

        # process this in steps, so that the ram usage is below the available ram
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
            # delete bad frames from data
            if self.filter_thres_bad_frames != 0 or self.filter_thres_mips != 0:
                data = an.exclude_mips_and_bad_frames(
                    data, self.filter_thres_mips, self.filter_thres_bad_frames
                )

            if self.filter_comm_mode is True:
                an.correct_common_mode(data)
            # calculate rndr signals and update file
            avg_over_nreps = utils.get_avg_over_nreps(data)
            io.add_array(
                self.analysis_file,
                "filter/precal/rndr_signals_after_common",
                avg_over_nreps,
            )
            # subtract fitted offset from data
            fitted_offset = io.get_data_from_file(
                self.analysis_file, "offnoi/fit/mean1"
            )
            avg_over_nreps -= fitted_offset
            io.add_array(
                self.analysis_file, "filter/rndr_signals/all_frames", avg_over_nreps
            )
            io.add_array(
                self.analysis_file,
                "filter/rndr_signals/average",
                np.nanmean(avg_over_nreps, axis=0),
            )
            # calculate bad slopes and update file
            if self.filter_thres_bad_slopes != 0:
                slopes = an.get_slopes(data)
                io.add_array(self.analysis_file, "filter/slopes/all_frames", slopes)
            self._logger.info(f"Finished step {step+1} of {steps_needed} total Steps")
            total_frames_processed += frames_per_step

        slopes = io.get_data_from_file(self.analysis_file, "filter/slopes/all_frames")
        bad_slopes_pos = np.full(slopes.shape, False, dtype=bool)
        bad_slopes_fit = np.zeros((6, self.column_size, self.row_size))

        # TODO: paralellize this
        # Calculate the bad slopes
        self._logger.info("Start calculating bad slopes")
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
                bad_slopes_fit[:, row, col] = fit_pixelwise

        io.add_array(self.analysis_file, "filter/slopes/bad_slopes_pos", bad_slopes_pos)
        io.add_array(
            self.analysis_file,
            "filter/slopes/bad_slopes_count",
            np.sum(bad_slopes_pos, axis=0),
        )
        io.add_array(self.analysis_file, "filter/slopes/bad_slopes_fit", bad_slopes_fit)
        self._logger.info("Finished calculating bad slopes")

        self._logger.info("Start calculating offset by fitting pixel wise")
        # load avg_over_nreps from the loop
        avg_over_nreps = io.get_data_from_file(
            self.analysis_file, "filter/rndr_signals/all_frames"
        )
        avg_over_nreps[bad_slopes_pos] = np.nan
        noise_map = io.get_data_from_file(self.analysis_file, "offnoi/fit/sigma1")
        io.add_array(
            self.analysis_file,
            "filter/rndr_signals/all_frames_slopes_removed",
            avg_over_nreps,
        )
        io.add_array(
            self.analysis_file,
            "filter/rndr_signals/average_slopes_removed",
            np.nanmean(avg_over_nreps, axis=0),
        )
        structure = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        event_array = an.group_pixels(
            avg_over_nreps,
            self.filter_thres_event_prim,
            self.filter_thres_event_sec,
            noise_map,
            structure,
        )
        io.add_array(self.analysis_file, "filter/events/event_array", event_array)
        io.add_array(
            self.analysis_file,
            "filter/events/event_count",
            np.sum(event_array != 0, axis=0),
        )
        fitted = fit.get_fit_over_frames(avg_over_nreps, peaks=2)
        io.add_array(self.analysis_file, "gain/fit/amplitude1", fitted[0, :, :])
        io.add_array(self.analysis_file, "gain/fit/mean1", fitted[1, :, :])
        io.add_array(self.analysis_file, "gain/fit/sigma1", fitted[2, :, :])
        io.add_array(self.analysis_file, "gain/fit/error_amplitude1", fitted[3, :, :])
        io.add_array(self.analysis_file, "gain/fit/error_mean1", fitted[4, :, :])
        io.add_array(self.analysis_file, "gain/fit/error_sigma1", fitted[5, :, :])
        io.add_array(self.analysis_file, "gain/fit/amplitude2", fitted[6, :, :])
        io.add_array(self.analysis_file, "gain/fit/mean2", fitted[7, :, :])
        io.add_array(self.analysis_file, "gain/fit/sigma2", fitted[8, :, :])
        io.add_array(self.analysis_file, "gain/fit/error_amplitude2", fitted[9, :, :])
        io.add_array(self.analysis_file, "gain/fit/error_mean2", fitted[10, :, :])
        io.add_array(self.analysis_file, "gain/fit/error_sigma2", fitted[11, :, :])
