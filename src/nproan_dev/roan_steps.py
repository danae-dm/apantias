import gc
import os
import psutil
from datetime import datetime

import numpy as np

from . import logger
from . import numba_funcs as af
from . import analysis as an
from . import params as pm
from . import fitting as fit
from . import parallel_funcs as pf

class RoanSteps():
    _logger = logger.Logger('nproan-RoanSteps', 'debug').get_logger()

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
        os.makedirs(self.step_dir, exist_ok=True)
        self._logger.info(f'Created working directory for offnoi step: {self.step_dir}')
        estimated_ram_usage = af.get_ram_usage_in_gb(
            self.offnoi_nframes, self.column_size, self.offnoi_nreps, self.row_size)*2.5
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
            data, self.current_offset = an.get_data_2(self.offnoi_bin_file[0], 
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
            #offset the data and correct for common mode if necessary
            data -= avg_over_frames[np.newaxis,:,:,:]
            if self.offnoi_comm_mode is True:
                an.correct_common_mode(data)
            #calculate rndr signals and update file
            avg_over_nreps = af.get_avg_over_nreps(data)
            self._update_npy_file('offnoi_rndr_signals.npy', avg_over_nreps)
            #calculate bad slopes and update file
            if self.offnoi_thres_bad_slopes != 0:
                slopes = an.get_bad_slopes(data, 
                                           self.offnoi_thres_bad_slopes,
                                           self.total_frames_processed)
                self._update_npy_file('offnoi_bad_slopes_slopes.npy', slopes)
            self.total_frames_processed += self.final_frames_per_step
            self._logger.info(f'Finished step {step+1} of {steps_needed} total Steps')
            
        slopes = np.load(os.path.join(self.step_dir, 'offnoi_bad_slopes_slopes.npy'))
        pos_list = []
        fit_params_list = []
        for row in range(slopes.shape[1]):
            for col in range(slopes.shape[2]):
                slopes_pixelwise = slopes[:,row,col]
                fit_pixelwise = fit.fit_gauss_to_hist(slopes_pixelwise.flatten())
                lower_bound = fit_pixelwise[1] - self.offnoi_thres_bad_slopes*np.abs(fit_pixelwise[2])
                upper_bound = fit_pixelwise[1] + self.offnoi_thres_bad_slopes*np.abs(fit_pixelwise[2])
                bad_slopes_mask = (slopes_pixelwise < lower_bound) | (slopes_pixelwise > upper_bound)
                frame = np.where(bad_slopes_mask)[0]
                row_array = np.full(frame.shape, row)
                col_array = np.full(frame.shape, col)
                pos_list.append(np.array([frame, row_array, col_array]).T)
                fit_params_list.append(fit_pixelwise)
        bad_slopes_pos = np.vstack(pos_list)
        bad_slopes_fit = np.vstack(fit_params_list)
        np.save(os.path.join(self.step_dir, 'offnoi_bad_slopes_pos.npy'), bad_slopes_pos)
        np.save(os.path.join(self.step_dir, 'offnoi_bad_slopes_fit.npy'), bad_slopes_fit)

        avg_over_nreps_final = np.load(os.path.join(self.step_dir, 
                                                    'offnoi_rndr_signals.npy'))
        bad_slopes_pos = np.load(os.path.join(self.step_dir, 'offnoi_bad_slopes_pos.npy'),
                                 allow_pickle=True)
        avg_over_nreps_slopes_removed = pf.set_values_to_nan(avg_over_nreps_final, bad_slopes_pos)
        self._logger.info('Fitting pixelwise for offset and noise')
        fitted = fit.get_fit_gauss(avg_over_nreps)
        fitted_slopes_removed = fit.get_fit_gauss(avg_over_nreps_slopes_removed)
        np.save(os.path.join(self.step_dir, 'offnoi_fit.npy'), fitted)
        np.save(os.path.join(self.step_dir, 'offnoi_fit_slopes_removed.npy'), fitted_slopes_removed)

    def calc_filter_step(self) -> None:
        self.step_dir = self.filter_dir
        #create the working directory for the filter step
        os.makedirs(self.step_dir, exist_ok=True)
        self._logger.info(f'Created working directory for filter step: {self.step_dir}')
        estimated_ram_usage = af.get_ram_usage_in_gb(
            self.filter_nframes, self.column_size, self.filter_nreps, self.row_size)*2.5
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
            data, self.current_offset = an.get_data_2(self.filter_bin_file[0], 
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
                slopes = an.get_bad_slopes(data, 
                                           self.filter_thres_bad_slopes,
                                           self.total_frames_processed)
                self._update_npy_file('filter_bad_slopes_slopes.npy', slopes)
            self._logger.info(f'Finished step {step+1} of {steps_needed} total Steps')
            self.total_frames_processed += self.final_frames_per_step

        slopes = np.load(os.path.join(self.step_dir, 'filter_bad_slopes_slopes.npy'))
        pos_list = []
        fit_params_list = []
        for row in range(slopes.shape[1]):
            for col in range(slopes.shape[2]):
                slopes_pixelwise = slopes[:,row,col]
                fit_pixelwise = fit.fit_gauss_to_hist(slopes_pixelwise.flatten())
                lower_bound = fit_pixelwise[1] - self.offnoi_thres_bad_slopes*np.abs(fit_pixelwise[2])
                upper_bound = fit_pixelwise[1] + self.offnoi_thres_bad_slopes*np.abs(fit_pixelwise[2])
                bad_slopes_mask = (slopes_pixelwise < lower_bound) | (slopes_pixelwise > upper_bound)
                frame = np.where(bad_slopes_mask)[0]
                row_array = np.full(frame.shape, row)
                col_array = np.full(frame.shape, col)
                pos_list.append(np.array([frame, row_array, col_array]).T)
                fit_params_list.append(fit_pixelwise)
        bad_slopes_pos = np.vstack(pos_list)
        bad_slopes_fit = np.vstack(fit_params_list)
        np.save(os.path.join(self.step_dir, 'filter_bad_slopes_pos.npy'), bad_slopes_pos)
        np.save(os.path.join(self.step_dir, 'filter_bad_slopes_fit.npy'), bad_slopes_fit)
        
        avg_over_nreps_final = np.load(os.path.join(self.step_dir, 'filter_rndr_signals.npy'))
        bad_slopes_pos = np.load(os.path.join(self.step_dir, 'filter_bad_slopes_pos.npy'),
                                 allow_pickle=True)
        avg_over_nreps_slopes_removed = pf.set_values_to_nan(avg_over_nreps_final, bad_slopes_pos)
        fitted_noise = np.load(os.path.join(self.offnoi_dir, 'offnoi_fit.npy'))[2]
        event_map = an.calc_event_map(avg_over_nreps_final,
                                      fitted_noise, 
                                      self.filter_thres_event)
        np.save(os.path.join(self.step_dir, 'event_map.npy'), event_map)
        np.save(os.path.join(self.step_dir, 'sum_of_event_signals.npy'),
                an.get_sum_of_event_signals(event_map, self.row_size, self.column_size))
        np.save(os.path.join(self.step_dir, 'sum_of_event_counts.npy'),
                an.get_sum_of_event_counts(event_map, self.row_size, self.column_size))
        event_map_slopes_removed = an.calc_event_map(avg_over_nreps_slopes_removed,
                                                        fitted_noise,
                                                        self.filter_thres_event)
        np.save(os.path.join(self.step_dir, 'event_map_slopes_removed.npy'), event_map_slopes_removed)
        np.save(os.path.join(self.step_dir, 'sum_of_event_signals_slopes_removed.npy'),
                an.get_sum_of_event_signals(event_map_slopes_removed, self.row_size, self.column_size))
        np.save(os.path.join(self.step_dir, 'sum_of_event_counts_slopes_removed.npy'),
                an.get_sum_of_event_counts(event_map_slopes_removed, self.row_size, self.column_size))