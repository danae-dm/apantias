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

"""
Planned structure of the analysis.h5 output file:
datasets: ~
groups: /
/1_offnoi
    /1_nrep_data
        ~signal_values
            # raw signals, averaged over nreps, after common mode correction
        ~slope_values
            # slope values (simple linear fit) of the raw signals
    /2_slopes
        ~slope_fit
            # slope values from precal are fitted pixel wise with a gaussian
        ~bad_slopes_mask
            # mask of bad slopes is calculated from the pixelwise fit and the threshold from the params file
        ~bad_slopes_count
            # count of number of bad slopes per pixel
        ~signal_values
            # raw signals after common mode correction, bad slopes are set to nan
    /3_outliers
        ~outliers_fit
            # signal values after common mode correction and bad slopes removed are fitted pixel wise with a gaussian
        ~outliers_mask
            # mask of outliers is calculated from the pixelwise fit and the threshold from the params file
        ~outliers_count
            # count of number of outliers per pixel
        ~signal_values
            # signal values after removing bad slopes and outliers
    /4_fit
        ~fit_1_peak
        # signal values after common mode correction, bad slopes removed and outliers removed are fitted pixel wise with a gaussian
        ~fit_2_peak
        # double gauss
    /5_final
        ~offset
            # offset value from the gaussian fit
        ~noise
            # noise value from the gaussian fit
        ~signal_values
            # raw signals after common mode correction, bad slopes removed, outliers removed and applied offset

/2_filter
    /1_nrep_data
        ~signal_values
            # raw signals, averaged over nreps, after common mode correction and offset from offnoi step subtracted
        ~slope_values
            # slope values (simple linear fit) of the raw signals
    /2_slopes
        ~slope_fit
            # slope values from precal are fitted pixel wise with a gaussian
        ~bad_slopes_mask
            # mask of bad slopes is calculated from the pixelwise fit and the threshold from the params file
        ~bad_slopes_count
            # count of number of bad slopes per pixel
        ~signal_values
            # raw signals after common mode correction, bad slopes are set to nan
        ~signal_values_offset_corrected
    /3_outliers
        ~outliers_fit
            # signal values after common mode correction and bad slopes removed are fitted pixel wise with a gaussian
        ~outliers_mask
            # mask of outliers is calculated from the pixelwise fit and the threshold from the params file
        ~outliers_count
            # count of number of outliers per pixel
        ~signal_values
            # signal values after removing bad slopes and outliers
    /4_events
        ~event_map
            # event map is calculated from the signal values, the noise values from the offnoi step and the thresholds from the params file
        ~event_map_counts
            # count of number of events per pixel
        ~event_details
            #TODO: implement pandas table with event details
        ~bleedthrough
            #TODO: implement bleedthrough calculation
/3_gain
    /fit_with_noise
        #TODO: Move simple 2 Gauss fit from filter step to here
    /signal_fit
        #TODO: somehow cut noise and fit a gaussian to the signal values

"""


class RoanSteps:
    _logger = logger.Logger("nproan-RoanSteps", "info").get_logger()

    def __init__(self, prm_file: str) -> None:
        self.prm_file = prm_file
        self.analysis_file_created = False
        self._logger.info(f"RoanSteps initialized with parameter file: {prm_file}")
        self._logger.info("To run the analysis steps, call:")
        self._logger.info("calc_offnoi_step()")
        self._logger.info("calc_filter_step()")
        self._logger.info("calc_gain_step()")
        self._logger.info(
            "If a external offsetmap or noisemap is used, make sure the path in the params file is set. And start with the filter step."
        )
        self._logger.info("")

    def load(self, prm_file: str) -> None:
        # load parameter file
        self.params = params.Params(prm_file)
        self.params_dict = self.params.get_dict()

        # polarity is from the old code, im not quite sure why it is -1
        self.polarity = -1

        # common parameters from params file
        self.results_dir = self.params_dict["common_results_dir"]
        self.available_cpus = self.params_dict["common_available_cpus"]
        self.mem_per_cpu_gb = self.params_dict["common_mem_per_cpu_gb"]
        self.attributes_dict = self.params_dict["common_attributes"]
        self.ram_available = self.available_cpus * self.mem_per_cpu_gb

        # offnoi parameters from params file
        self.offnoi_data_file = self.params_dict["offnoi_data_file"]
        self.offnoi_nframes_eval = self.params_dict["offnoi_nframes_eval"]
        self.offnoi_nreps_eval = self.params_dict["offnoi_nreps_eval"]
        self.offnoi_comm_mode = self.params_dict["offnoi_comm_mode"]
        self.offnoi_thres_bad_slopes = self.params_dict["offnoi_thres_bad_slopes"]

        # filter parameters from params file
        self.filter_data_file = self.params_dict["filter_data_file"]
        self.filter_nframes_eval = self.params_dict["filter_nframes_eval"]
        self.filter_nreps_eval = self.params_dict["filter_nreps_eval"]
        self.filter_comm_mode = self.params_dict["filter_comm_mode"]
        self.filter_thres_event_prim = self.params_dict["filter_thres_event_prim"]
        self.filter_thres_event_sec = self.params_dict["filter_thres_event_sec"]
        self.filter_thres_bad_slopes = self.params_dict["filter_thres_bad_slopes"]
        self.filter_ext_offsetmap = self.params_dict["filter_ext_offsetmap"]
        self.filter_ext_noisemap = self.params_dict["filter_ext_noisemap"]

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
        start = self.offnoi_nframes_eval[0]
        stop = self.offnoi_nframes_eval[1]
        step = self.offnoi_nframes_eval[2]
        if stop == -1:
            self.offnoi_nframes_eval[1] = self.offnoi_total_frames
            self.offnoi_nframes_slice = f"{start}:{self.offnoi_total_frames}:{step}"
        else:
            self.offnoi_nframes_slice = f"{start}:{stop}:{step}"
        start = self.offnoi_nreps_eval[0]
        stop = self.offnoi_nreps_eval[1]
        step = self.offnoi_nreps_eval[2]
        if stop == -1:
            self.offnoi_nreps_eval[1] = self.offnoi_total_nreps
            self.offnoi_nreps_slice = f"{start}:{self.offnoi_total_nreps}:{step}"
        else:
            self.offnoi_nreps_slice = f"{start}:{stop}:{step}"
        start = self.filter_nframes_eval[0]
        stop = self.filter_nframes_eval[1]
        step = self.filter_nframes_eval[2]
        if stop == -1:
            self.filter_nframes_eval[1] = self.filter_total_frames
            self.filter_nframes_slice = f"{start}:{self.filter_total_frames}:{step}"
        else:
            self.filter_nframes_slice = f"{start}:{stop}:{step}"
        start = self.filter_nreps_eval[0]
        stop = self.filter_nreps_eval[1]
        step = self.filter_nreps_eval[2]
        if stop == -1:
            self.filter_nreps_eval[1] = self.filter_total_nreps
            self.filter_nreps_slice = f"{start}:{self.filter_total_nreps}:{step}"
        else:
            self.filter_nreps_slice = f"{start}:{stop}:{step}"

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
            self.attributes_dict,
        )
        self.analysis_file_created = True
        self._logger.info(
            f"Created analysis h5 file: {self.results_dir}/{self.analysis_file_name}"
        )
        self._logger.info(f"Parameters loaded:")
        self.params.print_contents()

    def calc_offnoi_step(self) -> None:
        self.load(self.prm_file)
        estimated_ram_usage = (
            utils.get_ram_usage_in_gb(
                self.offnoi_nframes_eval,
                self.column_size,
                self.offnoi_nreps_eval,
                self.row_size,
            )
            * 2.5  # this is estimated, better safe than sorry
        )
        self._logger.info(f"\n")
        self._logger.info(f"---------Start offnoi step---------")
        self._logger.info(f"RAM available: {self.ram_available:.1f} GB")
        self._logger.info(f"Estimated RAM usage: {estimated_ram_usage:.1f} GB")
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f"Steps needed: {steps_needed}")

        # (planned) frames per step, so that ram usage is below the available ram
        frames_per_step = int(self.offnoi_nframes_eval / steps_needed)

        # total processed frames over all steps
        total_frames_processed = 0

        """
        Start of the loop to process the data in steps, so that the ram usage is below the available ram
        -removed mips and bad frames for now
        """
        for step in range(steps_needed):
            self._logger.info(f"Start processing step {step+1} of {steps_needed}")
            current_frame_slice = (
                f"{total_frames_processed}:{total_frames_processed + frames_per_step}"
            )
            slice = f"[{current_frame_slice},:,{self.offnoi_nreps_slice},:]"
            data = (
                io.get_data_from_file(self.offnoi_data_file, "data", slice)
                * self.polarity
            )
            self._logger.info(f"Data loaded: {data.shape}")
            if self.offnoi_comm_mode is True:
                an.correct_common_mode(data)
            avg_over_nreps = utils.get_avg_over_nreps(data)
            output_info = {
                "info": "raw signals, averaged over nreps, after common mode correction"
            }
            io.add_array_to_file(
                self.analysis_file,
                "1_offnoi/1_nrep_data/signal_values",
                avg_over_nreps,
                attributes=output_info,
            )
            if self.offnoi_thres_bad_slopes != 0:
                slopes = an.get_slopes(data)
                output_info = {
                    "info": "slope values (simple linear fit) of the raw signals"
                }
                io.add_array_to_file(
                    self.analysis_file,
                    "1_offnoi/1_nrep_data/slope_values",
                    slopes,
                    attributes=output_info,
                )
            total_frames_processed += frames_per_step
            self._logger.info(f"Finished step {step+1} of {steps_needed} total Steps")

        self._logger.info("Start calculating bad slopes")
        slopes = io.get_data_from_file(
            self.analysis_file, "1_offnoi/1_nrep_data/slope_values"
        )
        fitted = fit.get_pixelwise_fit(slopes, peaks=1)
        lower_bound = fitted[:, :, 1] - self.offnoi_thres_bad_slopes * np.abs(
            fitted[:, :, 2]
        )
        upper_bound = fitted[:, :, 1] + self.offnoi_thres_bad_slopes * np.abs(
            fitted[:, :, 2]
        )
        bad_slopes_mask = (slopes < lower_bound) | (slopes > upper_bound)
        output_info = {
            "info": "slope values from nrep_data step are fitted pixel wise with a gaussian"
        }
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/2_slopes/fit_parameters",
            fitted,
            attributes=output_info,
        )
        output_info = {
            "info": "mask of bad slopes is calculated from the pixelwise fit"
        }
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/2_slopes/bad_slopes_mask",
            bad_slopes_mask,
            attributes=output_info,
        )
        output_info = {"info": "count of number of bad slopes per pixel"}
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/2_slopes/bad_slopes_count",
            np.sum(bad_slopes_mask, axis=0),
            attributes=output_info,
        )
        failed_fits = np.sum(np.isnan(fitted[:, :, 1]))
        if failed_fits > 0:
            self._logger.warning(
                f"Failed fits: {failed_fits} ({failed_fits/(self.column_size*self.row_size)*100:.2f}%)"
            )

        # get avg_over_nreps from the loop
        avg_over_nreps = io.get_data_from_file(
            self.analysis_file, "1_offnoi/1_nrep_data/signal_values"
        )
        # set bad slopes to nan, so they not interfere in future calculations
        avg_over_nreps[bad_slopes_mask] = np.nan
        output_info = {
            "info": "raw signals after common mode correction, bad slopes are set to nan"
        }
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/2_slopes/signal_values",
            avg_over_nreps,
            attributes=output_info,
        )
        bad_signals = np.sum(bad_slopes_mask)
        self._logger.warning(
            f"Signals removed due to bad slopes: {bad_signals} ({bad_signals/(bad_slopes_mask.size)*100:.2f}%)"
        )
        self._logger.info("Finished calculating bad slopes")

        self._logger.info("Start preliminary fit to remove outliers")
        fitted = fit.get_pixelwise_fit(avg_over_nreps, peaks=1)
        output_info = {
            "info": "signal values after common mode correction and bad slopes removed are fitted pixel wise with a gaussian"
        }
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/3_outliers/fit_parameters",
            fitted,
            attributes=output_info,
        )
        lower_bound = fitted[:, :, 1] - 8 * fitted[:, :, 2]
        upper_bound = fitted[:, :, 1] + 8 * fitted[:, :, 2]
        prelim_fit_mask = (avg_over_nreps < lower_bound) | (
            avg_over_nreps > upper_bound
        )
        avg_over_nreps[prelim_fit_mask] = np.nan
        output_info = {"info": "signal values after removing outliers"}
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/3_outliers/signal_values",
            avg_over_nreps,
            attributes=output_info,
        )
        output_info = {"info": "mask of outliers is calculated from the pixelwise fit"}
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/3_outliers/outliers_mask",
            prelim_fit_mask,
            attributes=output_info,
        )
        output_info = {"info": "count of number of outliers per pixel"}
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/3_outliers/outliers_count",
            np.sum(prelim_fit_mask, axis=0),
            attributes=output_info,
        )
        failed_fits = np.sum(np.isnan(fitted[1, :, :]))
        if failed_fits > 0:
            self._logger.warning(
                f"Failed fits: {failed_fits} ({failed_fits/(self.column_size*self.row_size)*100:.2f}%)"
            )
        bad_signals = np.sum(prelim_fit_mask)
        self._logger.warning(
            f"Signals removed due to preliminary fit: {bad_signals} ({bad_signals/(prelim_fit_mask.size)*100:.2f}%)"
        )
        self._logger.info("Finished preliminary fit to remove outliers")

        self._logger.info("Start fitting 1 peak gaussian to determine offset")
        fitted = fit.get_pixelwise_fit(avg_over_nreps, peaks=1)
        output_info = {
            "info": "signal values after common mode correction, bad slopes removed and outliers removed are fitted pixel wise with a gaussian"
        }
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/4_fit/fit_parameters",
            fitted,
            attributes=output_info,
        )
        failed_fits = np.sum(np.isnan(fitted[:, :, 1]))
        if failed_fits > 0:
            self._logger.warning(
                f"Failed fits: {failed_fits} ({failed_fits/(self.column_size*self.row_size)*100:.2f}%)"
            )
        self._logger.info("Finished fitting 1 peak gaussian to determine offset")

        self._logger.info("Offset data and save rndr_signals")
        avg_over_nreps -= fitted[:, :, 1]
        output_info = {"info": "Data with signal offset from the gaussian fit"}
        io.add_array_to_file(
            self.analysis_file,
            "1_offnoi/5_final/signal_values",
            avg_over_nreps,
            attributes=output_info,
        )
        self._logger.info("Finished offsetting data and saving rndr_signals")
        self._logger.info("---------Finished offnoi step---------")

    def calc_filter_step(self) -> None:

        if not self.analysis_file_created:
            self.load(self.prm_file)
            if (self.filter_ext_noisemap or self.filter_ext_offsetmap) == "":
                self._logger.error(
                    "Offnoi step was not run, and no external offsetmap or noisemap provided."
                )
                raise ValueError(
                    "Offnoi step was not run, and no external offsetmap or noisemap provided."
                )

            else:
                try:
                    # if offset step was not run, load external offsetmap and noisemap
                    offnoi_offset = np.load(self.filter_ext_offsetmap)
                    noise_map = np.load(self.filter_ext_noisemap)
                except:
                    self._logger.error(
                        "External offsetmap or noisemap could not be loaded."
                    )
                    raise ValueError(
                        "External offsetmap or noisemap could not be loaded."
                    )
        else:
            # if offset step was run, load offsetmap and noisemap
            offnoi_offset = io.get_data_from_file(
                self.analysis_file, "1_offnoi/4_fit/fit_parameters", "[:,:,1]"
            )
            noise_map = io.get_data_from_file(
                self.analysis_file, "1_offnoi/4_fit/fit_parameters", "[:,:,2]"
            )
        estimated_ram_usage = (
            utils.get_ram_usage_in_gb(
                self.filter_nframes_eval,
                self.column_size,
                self.filter_nreps_eval,
                self.row_size,
            )
            * 2.5  # this is estimated, better safe than sorry
        )
        self._logger.info(f"\n")
        self._logger.info(f"---------Start filter step---------")
        self._logger.info(f"RAM available: {self.ram_available:.1f} GB")
        self._logger.info(f"Estimated RAM usage: {estimated_ram_usage:.1f} GB")
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f"Steps needed: {steps_needed}")

        # (planned) frames per step, so that ram usage is below the available ram
        frames_per_step = int(self.filter_nframes_eval / steps_needed)

        # total processed frames over all steps
        total_frames_processed = 0

        """
        Start of the loop to process the data in steps, so that the ram usage is below the available ram
        -removed mips and bad frames for now
        """
        for step in range(steps_needed):
            self._logger.info(f"Start processing step {step+1} of {steps_needed}")
            current_frame_slice = (
                f"{total_frames_processed}:{total_frames_processed + frames_per_step}"
            )
            slice = f"[{current_frame_slice},:,{self.filter_nreps_slice},:]"
            data = (
                io.get_data_from_file(self.filter_data_file, "data", slice)
                * self.polarity
            )
            self._logger.info(f"Data loaded: {data.shape}")
            if self.filter_comm_mode is True:
                an.correct_common_mode(data)
            avg_over_nreps = utils.get_avg_over_nreps(data)
            output_info = {
                "info": "raw signals, averaged over nreps, after common mode correction"
            }
            io.add_array_to_file(
                self.analysis_file,
                "2_filter/1_nrep_data/signal_values",
                avg_over_nreps,
                attributes=output_info,
            )
            if self.filter_thres_bad_slopes != 0:
                slopes = an.get_slopes(data)
                output_info = {
                    "info": "slope values (simple linear fit) of the raw signals"
                }
                io.add_array_to_file(
                    self.analysis_file,
                    "2_filter/1_nrep_data/slope_values",
                    slopes,
                    attributes=output_info,
                )
            total_frames_processed += frames_per_step
            self._logger.info(f"Finished step {step+1} of {steps_needed} total Steps")

        self._logger.info("Start calculating bad slopes")
        slopes = io.get_data_from_file(
            self.analysis_file, "2_filter/1_nrep_data/slope_values"
        )
        fitted = fit.get_pixelwise_fit(slopes, peaks=1)
        lower_bound = fitted[:, :, 1] - self.filter_thres_bad_slopes * np.abs(
            fitted[:, :, 2]
        )
        upper_bound = fitted[:, :, 1] + self.filter_thres_bad_slopes * np.abs(
            fitted[:, :, 2]
        )
        bad_slopes_mask = (slopes < lower_bound) | (slopes > upper_bound)
        output_info = {
            "info": "slope values from precal are fitted pixel wise with a gaussian"
        }
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/2_slopes/fit_parameters",
            fitted,
            attributes=output_info,
        )
        output_info = {
            "info": "mask of bad slopes is calculated from the pixelwise fit"
        }
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/2_slopes/bad_slopes_mask",
            bad_slopes_mask,
            attributes=output_info,
        )
        output_info = {"info": "count of number of bad slopes per pixel"}
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/2_slopes/bad_slopes_count",
            np.sum(bad_slopes_mask, axis=0),
            attributes=output_info,
        )
        failed_fits = np.sum(np.isnan(fitted[:, :, 1]))
        if failed_fits > 0:
            self._logger.warning(
                f"Failed fits: {failed_fits} ({failed_fits/(self.column_size*self.row_size)*100:.2f}%)"
            )

        # load avg_over_nreps from the loop
        avg_over_nreps = io.get_data_from_file(
            self.analysis_file, "2_filter/1_nrep_data/signal_values"
        )
        # subtract offset from offnoi step
        avg_over_nreps -= offnoi_offset
        output_info = {
            "info": "raw signals after common mode correction and offset correction"
        }
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/2_slopes/signal_values_offset_corrected",
            avg_over_nreps,
            attributes=output_info,
        )
        # set bad slopes to nan, so they not interfere in future calculations
        avg_over_nreps[bad_slopes_mask] = np.nan
        output_info = {
            "info": "raw signals after common mode correction and offset, bad slopes are set to nan"
        }
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/2_slopes/signal_values",
            avg_over_nreps,
            attributes=output_info,
        )
        bad_signals = np.sum(bad_slopes_mask)
        self._logger.warning(
            f"Signals removed due to bad slopes: {bad_signals} ({bad_signals/(bad_slopes_mask.size)*100:.2f}%)"
        )
        self._logger.info("Finished calculating bad slopes")

        self._logger.info("Start preliminary fit to remove outliers")
        fitted = fit.get_pixelwise_fit(avg_over_nreps, peaks=1)
        output_info = {
            "info": "signal values after common mode correction and offset, bad slopes removed are fitted pixel wise with a gaussian"
        }
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/3_outliers/fit_parameters",
            fitted,
            attributes=output_info,
        )
        lower_bound = fitted[:, :, 1] - 8 * fitted[:, :, 2]
        upper_bound = fitted[:, :, 1] + 8 * fitted[:, :, 2]
        prelim_fit_mask = (avg_over_nreps < lower_bound) | (
            avg_over_nreps > upper_bound
        )
        avg_over_nreps[prelim_fit_mask] = np.nan
        output_info = {"info": "mask of outliers is calculated from the pixelwise fit"}
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/3_outliers/outliers_mask",
            prelim_fit_mask,
            attributes=output_info,
        )
        output_info = {"info": "count of number of outliers per pixel"}
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/3_outliers/outliers_count",
            np.sum(prelim_fit_mask, axis=0),
            attributes=output_info,
        )
        output_info = {"info": "signal values after removing bad slopes and outliers"}
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/3_outliers/signal_values",
            avg_over_nreps,
            attributes=output_info,
        )
        failed_fits = np.sum(np.isnan(fitted[1, :, :]))
        if failed_fits > 0:
            self._logger.warning(
                f"Failed fits: {failed_fits} ({failed_fits/(self.column_size*self.row_size)*100:.2f}%)"
            )
        bad_signals = np.sum(prelim_fit_mask)
        self._logger.warning(
            f"Signals removed due to preliminary fit: {bad_signals} ({bad_signals/(prelim_fit_mask.size)*100:.2f}%)"
        )
        self._logger.info("Finished preliminary fit to remove outliers")

        self._logger.info("Start Calculating event_map")
        structure = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        event_map = an.group_pixels(
            avg_over_nreps,
            self.filter_thres_event_prim,
            self.filter_thres_event_sec,
            noise_map,
            structure,
        )
        event_counts = event_map > 0
        event_counts_sum = np.sum(event_counts, axis=0)
        output_info = {
            "info": "event map is calculated from the signal values, events are grouped. 0 means no event."
        }
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/4_events/event_map",
            event_map,
            attributes=output_info,
        )
        output_info = {"info": "count of number of events per pixel and per frame"}
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/4_events/event_map_counts",
            event_counts,
            attributes=output_info,
        )
        output_info = {
            "info": "sum of counts of number of events per pixel and per frame"
        }
        io.add_array_to_file(
            self.analysis_file,
            "2_filter/4_events/event_map_counts_sum",
            event_counts_sum,
            attributes=output_info,
        )
        self._logger.info("Finished calculating event_map")
        self._logger.info("---------Finished filter step---------")

    def calc_gain_step(self) -> None:
        self._logger.info(f"\n")
        self._logger.info("---------Start gain step---------")
        self._logger.info("Start fitting 1 peak gaussian for gain calculation")
        avg_over_nreps = io.get_data_from_file(
            self.analysis_file,
            "2_filter/3_outliers/signal_values",
        )
        fitted = fit.get_pixelwise_fit(avg_over_nreps, peaks=2)
        output_info = {"info": "simple 2 peak gauss fit to determine gain"}
        io.add_array_to_file(
            self.analysis_file, "3_gain/fit_parameters", fitted, attributes=output_info
        )
        self._logger.info("Finished fitting 1 peak gaussian for gain calculation")
        self._logger.info("---------Finished gain step---------")
