import gc
import os
import psutil
from datetime import datetime

import numpy as np

from . import logger
from . import analysis_funcs as af
from . import analysis as an
from . import params as pm
from . import fitting as fit

class Steps():
    _logger = logger.Logger('nproan-preprocess', 'debug').get_logger()

    def __init__(self, prm_file: str, ram: int) -> None:
        self.ram_available = ram
        self.load(prm_file)

    def load(self, prm_file: str) -> None:
        #load parameter file
        self.params = pm.Params(prm_file)
        self.params_dict = self.params.get_dict()

        #common parameters
        self.results_dir = self.params_dict['common_results_dir']
        self.column_size = self.params_dict['common_column_size']
        self.row_size = self.params_dict['common_row_size']
        self.key_ints = self.params_dict['common_key_ints']
        self.bad_pixels = self.params_dict['common_bad_pixels']

        #offnoi parameters
        self.offnoi_bin_file = self.params_dict['offnoi_bin_file']
        self.offnoi_nreps = self.params_dict['offnoi_nreps']
        self.offnoi_nframes = self.params_dict['offnoi_nframes']
        self.offnoi_nreps_eval = self.params_dict['offnoi_nreps_eval']
        self.offnoi_comm_mode = self.params_dict['offnoi_comm_mode']
        self.offnoi_thres_mips = self.params_dict['offnoi_thres_mips']
        self.offnoi_thres_bad_frames = self.params_dict['offnoi_thres_bad_frames']
        self.offnoi_thres_bad_slopes = self.params_dict['offnoi_thres_bad_slopes']

        #filter parameters
        self.filter_bin_file = self.params_dict['filter_bin_file']
        self.filter_nreps = self.params_dict['filter_nreps']
        self.filter_nframes = self.params_dict['filter_nframes']
        self.filter_nreps_eval = self.params_dict['filter_nreps_eval']
        self.filter_comm_mode = self.params_dict['filter_comm_mode']
        self.filter_thres_mips = self.params_dict['filter_thres_mips']
        self.filter_thres_event = self.params_dict['filter_thres_event']
        self.filter_use_fitted_offset = self.params_dict['filter_use_fitted_offset']
        self.filter_thres_bad_frames = self.params_dict['filter_thres_bad_frames']
        self.filter_thres_bad_slopes = self.params_dict['filter_thres_bad_slopes']

        #class variables for tracking of preprocessing steps
        self.current_offset = 0
        self.total_frames_processed = 0
        self.frames_per_step = 0
        self.final_frames_per_step = 0

        #directories
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        #TODO: think of a better file name
        filename = os.path.basename(self.filter_bin_file[0])[:-4]
        self.common_dir = os.path.join(
            self.results_dir, timestamp + '_' + filename)
        #current step directory (is set in calc())
        self.step_dir = None
        self.offnoi_dir = os.path.join(self.common_dir, 'offnoi')
        self.filter_dir = os.path.join(self.common_dir, 'filter')
        self._logger.info(f'Parameters loaded:')
        self.params.print_contents()

    def _update_offset_raw(self, file_name: str, new_data: np.ndarray) -> None:
        file_path = os.path.join(self.offnoi_dir, file_name)
        if os.path.exists(file_path):
            old_data = np.load(file_path)
            output = (old_data * self.total_frames_processed 
                    + new_data * self.final_frames_per_step) / (self.total_frames_processed + self.final_frames_per_step)
        else:
            output = new_data
        np.save(file_path, output)

    def _update_npy_file(self, file_name: str, new_data: np.ndarray) -> None:
        file_path = os.path.join(self.step_dir, file_name)
        if os.path.exists(file_path):
            old_data = np.load(file_path)
            output = np.concatenate((old_data, new_data), axis=0)
        else:
            output = new_data
        np.save(file_path, output)

    def calc_offnoi_step(self) -> None:
        self.step_dir = self.offnoi_dir
        #create the working directory for the offnoi step
        os.makedirs(self.offnoi_dir, exist_ok=True)
        self._logger.info(f'Created working directory for offnoi step: {self.offnoi_dir}')
        estimated_ram_usage = af.get_ram_usage_in_gb(
            self.offnoi_nframes, self.column_size, self.offnoi_nreps, self.row_size)*2.3
        self._logger.info(f'RAM available: {self.ram_available:.1f} GB')
        self._logger.info(f'Estimated RAM usage: {estimated_ram_usage:.1f} GB')
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f'Steps needed: {steps_needed}')

        #set class variables
        #the initial offset for reading the bin file, first 8 bytes are header
        self.current_offset = 8
        #total processed frames over all steps
        self.total_frames_processed = 0
        #(planned) frames per step, so that ram usage is below the available ram
        self.frames_per_step = int(self.offnoi_nframes / steps_needed)
        #final frames per step, this is the actual number of frames processed 
        #in a step (after deleting bad frames)
        self.final_frames_per_step = 0

        for step in range(steps_needed):
            self._logger.info(f'Performing step {step+1} of {steps_needed} total Steps')
            #load data from bin file sequentially
            data, self.current_offset = an.get_data_2(self.offnoi_bin_file, 
                                         self.column_size, 
                                         self.row_size, 
                                         self.key_ints, 
                                         self.offnoi_nreps, 
                                         self.frames_per_step, 
                                         self.current_offset)
            
            #exclude nreps_eval from data
            if self.offnoi_nreps_eval:
                data = an.exclude_nreps_eval(data, self.offnoi_nreps_eval)
                self._logger.debug(f'Shape of data: {data.shape}')
            #set values of all frames and nreps of bad pixels to nan
            if self.bad_pixels:
                data = an.set_bad_pixellist_to_nan(data, self.bad_pixels)
            #delete bad frames from data
            if self.offnoi_thres_bad_frames != 0 or self.offnoi_thres_mips != 0:
                data = an.exclude_mips_and_bad_frames(data, self.offnoi_thres_mips, 
                                                      self.offnoi_thres_bad_frames)
                self._logger.debug(f'Shape of data: {data.shape}')

            self.final_frames_per_step = data.shape[0]
            #Calculate offset_raw on the raw data and update file
            avg_over_frames = af.get_avg_over_frames(data)
            self._update_offset_raw('offnoi_offset_raw.npy', avg_over_frames)
            self.total_frames_processed += self.final_frames_per_step
            #offset the data and correct for common mode if necessary
            data -= avg_over_frames[np.newaxis,:,:,:]
            if self.offnoi_comm_mode is True:
                an.correct_common_mode(data)
            #calculate rndr signals and update file
            avg_over_nreps = af.get_avg_over_nreps(data)
            self._update_npy_file('offnoi_rndr_signals.npy', avg_over_nreps)
            #TODO: rewrite slopes to get treshold from pixelwise fit
            #calculate bad slopes and update file
            if self.offnoi_thres_bad_slopes != 0:
                slopes_0, slopes_1, slopes_2 = an.get_bad_slopes(data, 
                                                                self.offnoi_thres_bad_slopes)
                self._update_npy_file('offnoi_bad_slopes_pos.npy', slopes_0)
                self._update_npy_file('offnoi_bad_slopes_data.npy', slopes_1)
                self._update_npy_file('offnoi_bad_slopes_value.npy', slopes_2)
            self._logger.info(f'Finished step {step+1} of {steps_needed} total Steps')
        
        #TODO: Exclude bad slopes from fit
        self._logger.info('Fitting pixelwise for offset and noise')
        fitted = fit.get_fit_gauss(avg_over_nreps)
        np.save(os.path.join(self.offnoi_dir, 'offnoi_fit.npy'), fitted)

    def calc_filter_step(self) -> None:
        self.step_dir = self.filter_dir
        #create the working directory for the filter step
        os.makedirs(self.filter_dir, exist_ok=True)
        self._logger.info(f'Created working directory for filter step: {self.filter_dir}')
        estimated_ram_usage = af.get_ram_usage_in_gb(
            self.filter_nframes, self.column_size, self.filter_nreps, self.row_size)*2.3
        self._logger.info(f'RAM available: {self.ram_available:.1f} GB')
        self._logger.info(f'Estimated RAM usage: {estimated_ram_usage:.1f} GB')
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f'Steps needed: {steps_needed}')

        #set class variables
        #the initial offset for reading the bin file, first 8 bytes are header
        self.current_offset = 8
        #total processed frames over all steps
        self.total_frames_processed = 0
        #(planned) frames per step, so that ram usage is below the available ram
        self.frames_per_step = int(self.filter_nframes / steps_needed)
        #final frames per step, this is the actual number of frames processed 
        #in a step (after deleting bad frames)
        self.final_frames_per_step = 0

        for step in range(steps_needed):
            self._logger.info(f'Performing step {step+1} of {steps_needed} total Steps')
            data, self.current_offset = an.get_data_2(self.filter_bin_file, 
                                         self.column_size, 
                                         self.row_size, 
                                         self.key_ints, 
                                         self.filter_nreps, 
                                         self.frames_per_step, 
                                         self.current_offset)
            
            #exclude nreps_eval from data
            if self.filter_nreps_eval:
                data = an.exclude_nreps_eval(data, self.filter_nreps_eval)
                self._logger.debug(f'Shape of data: {data.shape}')
            #set values of all frames and nreps of bad pixels to nan
            if self.bad_pixels:
                data = an.set_bad_pixellist_to_nan(data, self.bad_pixels)
            #delete bad frames from data
            if self.filter_thres_bad_frames != 0 or self.filter_thres_mips != 0:
                data = an.exclude_mips_and_bad_frames(data, self.filter_thres_mips,
                                                        self.filter_thres_bad_frames)
                self._logger.debug(f'Shape of data: {data.shape}')
            
            self.final_frames_per_step = data.shape[0]
            #Get offset_raw from offnoi step
            avg_over_frames = np.load(os.path.join(self.offnoi_dir, 'offnoi_offset_raw.npy'))
            #offset the data and correct for common mode if necessary
            data -= avg_over_frames[np.newaxis,:,:,:]
            if self.filter_comm_mode is True:
                an.correct_common_mode(data)
            #calculate rndr signals and update file
            avg_over_nreps = af.get_avg_over_nreps(data)
            # subtract fitted offset from data
            fitted_offset = np.load(os.path.join(self.offnoi_dir, 'offnoi_fit.npy'))[1]
            avg_over_nreps -= fitted_offset
            self._update_npy_file('filter_rndr_signals.npy', avg_over_nreps)
            #calculate bad slopes and update file
            if self.filter_thres_bad_slopes != 0:
                pos, data, value = an.get_bad_slopes(data, 
                                                     self.filter_thres_bad_slopes)
                self._update_npy_file('filter_bad_slopes_pos.npy', pos)
                self._update_npy_file('filter_bad_slopes_data.npy', data)
                self._update_npy_file('filter_bad_slopes_value.npy', value)
            self._logger.info(f'Finished step {step+1} of {steps_needed} total Steps')
        
        final_avg_over_nreps = np.load(os.path.join(self.filter_dir, 'filter_rndr_signals.npy'))
        fitted_noise = np.load(os.path.join(self.offnoi_dir, 'offnoi_fit.npy'))[2]
        event_map = an.calc_event_map(final_avg_over_nreps,
                                      fitted_noise, 
                                      self.filter_thres_event)
        np.save(os.path.join(self.filter_dir, 'event_map.npy'), event_map)
        np.save(os.path.join(self.filter_dir, 'sum_of_event_signals.npy'),
                an.get_sum_of_event_signals(event_map, self.row_size, self.column_size))
        np.save(os.path.join(self.filter_dir, 'sum_of_event_counts.npy'),
                an.get_sum_of_event_counts(event_map, self.row_size, self.column_size))