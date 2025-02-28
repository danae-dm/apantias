import gc
import os
import psutil
from datetime import datetime

import numpy as np

from . import utils
from . import analysis as an
from . import params
from . import fitting as fit
from . import file_io as io


from .logger import global_logger

_logger = global_logger
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


class Analysis:

    def __init__(self, prm_file: str) -> None:
        self.prm_file = prm_file
        self.params = params.Params(prm_file)
        _logger.info(f"APANTIAS Instance initialized with parameter file: {prm_file}")
        self.params_dict = self.params.print_contents()
        _logger.info("")
        # load values of parameter file
        self.params_dict = self.params.get_dict()
        self.results_dir = self.params_dict["results_dir"]
        self.data_h5 = self.params_dict["data_h5_file"]
        self.darkframe_dset = self.params_dict["darkframe_dset"]
        self.available_cpus = self.params_dict["available_cpus"]
        self.available_ram_gb = self.params_dict["available_ram_gb"]
        self.custom_attributes = self.params_dict["custom_attributes"]
        self.nframes_eval = self.params_dict["nframes_eval"]
        self.nreps_eval = self.params_dict["nreps_eval"]
        self.thres_bad_slopes = self.params_dict["thres_bad_slopes"]
        self.thres_event_prim = self.params_dict["thres_event_prim"]
        self.thres_event_sec = self.params_dict["thres_event_sec"]
        self.ext_offsetmap = self.params_dict["ext_offsetmap"]
        self.ext_noisemap = self.params_dict["ext_noisemap"]
        self.polarity = self.params_dict["polarity"]

        # get parameters from data_h5 file
        self.total_frames, self.column_size, self.row_size, self.nreps = (
            io._get_params_from_data_file(self.data_h5)
        )

        # create analysis h5 file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        bin_filename = os.path.basename(self.data_h5)[:-3]
        self.out_h5_name = f"{timestamp}_{bin_filename}.h5"
        self.out_h5 = os.path.join(self.results_dir, self.out_h5_name)
        io._create_analysis_file(
            self.results_dir,
            self.out_h5_name,
            self.params_dict,
            self.custom_attributes,
        )
        _logger.info(f"Created analysis h5 file: {self.results_dir}/{self.out_h5_name}")


class Default(Analysis):
    # inherits from Analysis
    def __init__(self, prm_file: str) -> None:
        super().__init__(prm_file)
        _logger.info("Default analysis initialized")

    def calculate(self):
        _logger.info("Start calculating bad slopes map")
        slopes = io.get_data_from_file(self.data_h5, "preproc_slope_nreps")
        fitted = utils.apply_pixelwise(
            slopes, fit.fit_gauss_to_hist, self.available_cpus
        )
        print(fitted.shape)
        _logger.info("Finished fitting")
        lower_bound = fitted[1, :, :] - self.thres_bad_slopes * np.abs(fitted[2, :, :])
        upper_bound = fitted[1, :, :] + self.thres_bad_slopes * np.abs(fitted[2, :, :])
        bad_slopes_mask = (slopes < lower_bound) | (slopes > upper_bound)
        output_info = {
            "info": "slope values from nrep_data step are fitted pixel wise with a gaussian"
        }
        io.add_array(
            self.out_h5, fitted, "1_slopes/fit_parameters", attributes=output_info
        )
        output_info = {
            "info": "mask of bad slopes is calculated from the pixelwise fit"
        }
        io.add_array(
            self.out_h5,
            bad_slopes_mask,
            "1_slopes/bad_slopes_mask",
            attributes=output_info,
        )
        output_info = {"info": "count of number of bad slopes per pixel"}
        io.add_array(
            self.out_h5,
            np.sum(bad_slopes_mask, axis=0),
            "1_slopes/bad_slopes_count",
            attributes=output_info,
        )
        failed_fits = np.sum(np.isnan(fitted[:, :, 1]))
        if failed_fits > 0:
            _logger.warning(
                f"Failed fits: {failed_fits} ({failed_fits/(self.column_size*self.row_size)*100:.2f}%)"
            )
