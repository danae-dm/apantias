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

class PreprocessData():
    _logger = logger.Logger('nproan-preprocess', 'debug').get_logger()

    def __init__(self, prm_file: str, ram_available: int) -> None:
        self.ram_available = ram_available
        self.load(prm_file)
        self._logger.info('PreprocessData object created')

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
        self.current_offset = 8
        self.total_frames_processed = 0
        self.frames_per_step = 0
        self.final_frames_per_step = 0


        #directories
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        #TODO: think of a better file name
        filename = os.path.basename(self.filter_bin_file[0])[:-4]
        self.common_dir = os.path.join(
            self.results_dir, timestamp + '_' + filename)
        self.step_dir = os.path.join(self.common_dir, 'preprocess')
        self._logger.info(f'Parameters loaded:')
        self.params.print_contents()

    def update_offset_raw(self, file_name: str, new_data: np.ndarray) -> None:
        file_path = os.path.join(self.step_dir, file_name)
        if os.path.exists(file_path):
            old_data = np.load(file_path)
            output = (old_data * self.total_frames_processed 
                    + new_data * self.final_frames_per_step) / (self.total_frames_processed + self.final_frames_per_step)
        else:
            output = new_data
        np.save(file_path, output)

    def update_npy_file(self, file_name: str, new_data: np.ndarray) -> None:
        file_path = os.path.join(self.step_dir, file_name)
        if os.path.exists(file_path):
            old_data = np.load(file_path)
            output = np.concatenate((old_data, new_data), axis=0)
        else:
            output = new_data
        np.save(file_path, output)

    def calculate(self) -> None:
        #create the directory for the data
        os.makedirs(self.common_dir, exist_ok=True)
        self._logger.info(f'Created common directory for data: {self.common_dir}')
        #now, create the working directory for the preproc step
        os.makedirs(self.step_dir, exist_ok=True)

        #PREPROCESS OFFNOI DATA

        #check how big the raw data will be
        self._logger.info('Start preprocessing offnoi data')
        #the estimation factor is estimated, adjust this if needed
        estimated_ram_usage = af.get_ram_usage_in_gb(
            self.offnoi_nframes, self.column_size, self.offnoi_nreps, self.row_size)*2.3
        # Get the available memory in bytes
        """
        this does not work in juyperhub
        virtual_memory = psutil.virtual_memory()
        available_memory_in_bytes = virtual_memory.available
        available_memory_in_gb = available_memory_in_bytes / (1024 ** 3)
        """

        self._logger.info(f'RAM available: {self.ram_available:.1f} GB')
        self._logger.info(f'Estimated RAM usage: {estimated_ram_usage:.1f} GB')
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f'Steps needed: {steps_needed}')

        #is set to 8, this is the offset (header) in the .bin file
        #calculate how much frames can be loaded at once:
        self.frames_per_step = int(self.offnoi_nframes / steps_needed)
        for step in range(steps_needed):
            self._logger.info(f'Performing step {step+1} of {steps_needed} total Steps')
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
            self.update_offset_raw('offnoi_offset_raw.npy', avg_over_frames)
            self.total_frames_processed += self.final_frames_per_step
            #calculate offset and update file
            avg_over_frames_and_nreps = af.get_avg_over_frames_and_nreps(data, avg_over_frames = avg_over_frames)
            self.update_npy_file('offnoi_offset.npy', avg_over_frames_and_nreps)
            #offset the data and correct for common mode if necessary
            data -= avg_over_frames[np.newaxis,:,:,:]
            if self.offnoi_comm_mode is True:
                an.correct_common_mode(data)
            #calculate rndr signals and update file
            avg_over_nreps = af.get_avg_over_nreps(data)
            self.update_npy_file('offnoi_rndr_signals.npy', avg_over_nreps)
            #calculate bad slopes and update file
            if self.offnoi_thres_bad_slopes != 0:
                bad_slopes_0, bad_slopes_1, bad_slopes_2 = an.get_bad_slopes(data, 
                                                                             self.offnoi_thres_bad_slopes)
                self.update_npy_file('offnoi_bad_slopes_pos.npy', bad_slopes_0)
                self.update_npy_file('offnoi_bad_slopes_data.npy', bad_slopes_1)
                self.update_npy_file('offnoi_bad_slopes_value.npy', bad_slopes_2)
            self._logger.info(f'Finished step {step+1} of {steps_needed} total Steps for offnoi data')

        #PREPROCESS FILTER DATA
        self._logger.info('Start preprocessing filter data')
        #check how big the raw data will be
        #the estimation factor is estimated, adjust this if needed
        estimated_ram_usage = af.get_ram_usage_in_gb(
            self.filter_nframes, self.column_size, self.filter_nreps, self.row_size)*2.3
        
        # Get the available memory in bytes
        '''
        this does not work in juyperhub
        virtual_memory = psutil.virtual_memory()
        available_memory_in_bytes = virtual_memory.available
        available_memory_in_gb = available_memory_in_bytes / (1024 ** 3)
        '''
        #just assume we have 64 GB of RAM
        available_memory_in_gb = 64

        self._logger.info(f'RAM available: {self.ram_available:.1f} GB')
        self._logger.info(f'Estimated RAM usage: {estimated_ram_usage:.1f} GB')
        steps_needed = int(estimated_ram_usage / self.ram_available) + 1
        self._logger.info(f'Steps needed: {steps_needed}')

        #calculate how much frames can be loaded at once, reset other counters
        self.frames_per_step = int(self.filter_nframes / steps_needed)
        self.current_offset = 8
        self.total_frames_processed = 0
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
            #Calculate offset_raw on the raw data and update file
            avg_over_frames = af.get_avg_over_frames(data)
            self.update_offset_raw('filter_offset_raw.npy', avg_over_frames)
            self.total_frames_processed += self.final_frames_per_step
            #calculate offset and update file
            avg_over_frames_and_nreps = af.get_avg_over_frames_and_nreps(data, avg_over_frames = avg_over_frames)
            self.update_npy_file('filter_offset.npy', avg_over_frames_and_nreps)
            #offset the data and correct for common mode if necessary
            data -= avg_over_frames[np.newaxis,:,:,:]
            if self.filter_comm_mode is True:
                an.correct_common_mode(data)
            #calculate rndr signals and update file
            avg_over_nreps = af.get_avg_over_nreps(data)
            self.update_npy_file('filter_rndr_signals.npy', avg_over_nreps)
            #calculate bad slopes and update file
            if self.filter_thres_bad_slopes != 0:
                bad_slopes_0, bad_slopes_1, bad_slopes_2 = an.get_bad_slopes(data, 
                                                                             self.filter_thres_bad_slopes)
                self.update_npy_file('filter_bad_slopes_pos.npy', bad_slopes_0)
                self.update_npy_file('filter_bad_slopes_data.npy', bad_slopes_1)
                self.update_npy_file('filter_bad_slopes_value.npy', bad_slopes_2)
            self._logger.info(f'Finished step {step+1} of {steps_needed} total Steps for offnoi data')


class OffNoi():
    _logger = logger.Logger('nproan-offnoi', 'debug').get_logger()

    def __init__(self, prm_file: str = None, preproc_dir: str = None) -> None:
        if prm_file is None or preproc_dir is None:
            raise ValueError('No parameter file or preprocess directory given.')
        self.load(prm_file, preproc_dir)
        self._logger.info('OffNoi object created')

    def load(self, prm_file: str, preproc_dir: str) -> None:
        self.params = pm.Params(prm_file)
        self.params_dict = self.params.get_dict()

        #common parameters
        self.results_dir = self.params_dict['common_results_dir']
        self.column_size = self.params_dict['common_column_size']
        self.row_size = self.params_dict['common_row_size']
        self.key_ints = self.params_dict['common_key_ints']
        self.bad_pixels = self.params_dict['common_bad_pixels']

        #offnoi parameters
        self.bin_file = self.params_dict['offnoi_bin_file']
        self.nreps = self.params_dict['offnoi_nreps']
        self.nframes = self.params_dict['offnoi_nframes']
        self.nreps_eval = self.params_dict['offnoi_nreps_eval']
        self.comm_mode = self.params_dict['offnoi_comm_mode']
        self.thres_mips = self.params_dict['offnoi_thres_mips']
        self.thres_bad_frames = self.params_dict['offnoi_thres_bad_frames']
        self.thres_bad_slopes = self.params_dict['offnoi_thres_bad_slopes']

        #directories
        #set self.common_dir to the parent directory of offnoi_dir
        self.common_dir = os.path.dirname(preproc_dir)
        self.step_dir = None

        #directories, they will be created in calculate()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = os.path.basename(self.bin_file)[:-4]
        self.common_dir = os.path.join(
            self.results_dir, timestamp + '_' + filename)
        self.step_dir = os.path.join(self.common_dir, 
            f'offnoi_{self.nreps}reps_{self.nframes}frames')

        self._logger.info(f'Parameters loaded:')
        self.params.print_contents()

    def calculate(self) -> None:   
        #now, create the working directory for the offnoi step
        os.makedirs(self.step_dir, exist_ok=True)
        self._logger.info(f'Created working directory for offnoi step: {self.step_dir}')
        # and save the parameter file there
        self.params.save(os.path.join(self.step_dir, 'parameters.json'))
        #get data from preprocess step
        avg_over_nreps = np.load(os.path.join(self.common_dir, 'preprocess', 'offnoi_rndr_signals.npy'))
        #calculate fitted offset and noise and save it (including fit errors)
        fit_curve_fit = fit.get_fit_gauss(avg_over_nreps)
        np.save(os.path.join(self.step_dir, 'offnoi_fit.npy'), fit_curve_fit)

class Filter():

    _logger = logger.Logger('nproan-filter', 'debug').get_logger()

    def __init__(self, prm_file: str = None, offnoi_dir: str = None) -> None:
        if prm_file is None or offnoi_dir is None:
            raise ValueError('No parameter file or offnoi_directory given.')
        self.load(prm_file, offnoi_dir)
        self._logger.info('Filter object created')

    def load(self, prm_file:str, offnoi_dir:str) -> None:
        self.params = pm.Params(prm_file)
        self.params_dict = self.params.get_dict()
        #common parameters
        self.results_dir = self.params_dict['common_results_dir']
        self.column_size = self.params_dict['common_column_size']
        self.row_size = self.params_dict['common_row_size']
        self.key_ints = self.params_dict['common_key_ints']
        self.bad_pixels = self.params_dict['common_bad_pixels']
        #filter parameters
        self.bin_file = self.params_dict['filter_bin_file']
        self.nreps = self.params_dict['filter_nreps']
        self.nframes = self.params_dict['filter_nframes']
        self.nreps_eval = self.params_dict['filter_nreps_eval']
        self.comm_mode = self.params_dict['filter_comm_mode']
        self.thres_mips = self.params_dict['filter_thres_mips']
        self.thres_event = self.params_dict['filter_thres_event']
        self.use_fitted_offset = self.params_dict['filter_use_fitted_offset']
        self.thres_bad_frames = self.params_dict['filter_thres_bad_frames']
        self.thres_bad_slopes = self.params_dict['filter_thres_bad_slopes']

        #directories
        #set self.common_dir to the parent directory of offnoi_dir
        self.common_dir = os.path.dirname(offnoi_dir)
        self.step_dir = None
        
        self._logger.info(f'Parameters loaded:')
        self.params.print_contents()
        
        self._logger.info('Checking parameters in offnoi directory')
        #look for a json file in the offnoi directory 
        if (not self.params.same_common_params(offnoi_dir)) \
            or (not self.params.same_offnoi_params(offnoi_dir)):
            self._logger.error('Parameters in offnoi directory do not match')
            return
        try:
            self.offset_fitted = af.get_array_from_file(
                offnoi_dir, 'offnoi_fit.npy'
            )[1]
            if self.offset_fitted is None:
                self._logger.error('Error loading fitted_offset data\n')
                return
            self.noise_fitted = af.get_array_from_file(
                offnoi_dir, 'offnoi_fit.npy'
            )[2]
            if self.noise_fitted is None:
                self._logger.error('Error loading fitted_noise data\n')
                return
            self.offnoi_dir = offnoi_dir
            self.common_dir = os.path.dirname(offnoi_dir)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.step_dir = os.path.join(
                self.common_dir,f'filter_{self.thres_event}_threshold'
            )
            self._logger.debug(self.step_dir)
        except:
            raise ValueError('Error loading offnoi data\n')

    def calculate(self) -> None:
        #create the working directory for the filter step
        os.makedirs(self.step_dir, exist_ok=True)
        self._logger.info(f'Created directory for filter step: {self.step_dir}')
        # and save the parameter file there
        self.params.save(os.path.join(self.step_dir, 'parameters.json'))
        avg_over_nreps = np.load(os.path.join(self.common_dir, 'preprocess', 'filter_rndr_signals.npy'))
        #subtract the fitted offset from the data
        avg_over_nreps -= self.offset_fitted
        event_map = an.calc_event_map(avg_over_nreps, self.noise_fitted, self.thres_event)
        np.save(os.path.join(self.step_dir, 'event_map.npy'),
                event_map)
        np.save(os.path.join(self.step_dir, 'sum_of_event_signals.npy'),
                an.get_sum_of_event_signals(event_map, self.row_size, self.column_size))
        np.save(os.path.join(self.step_dir, 'sum_of_event_counts.npy'),
                an.get_sum_of_event_counts(event_map, self.row_size, self.column_size))

class Gain():

    _logger = logger.Logger('nproan-gain', 'debug').get_logger()

    def __init__(self, prm_file: str = None, filter_dir: str = None) -> None:
        if prm_file is None or filter_dir is None:
            raise ValueError('No parameter file or filter directory given.')
        self.load(prm_file, filter_dir)
        self._logger.info('Gain object created')

    def load(self, prm_file: str, filter_dir: str) -> None:
        self.params = pm.Params(prm_file)
        self.params_dict = self.params.get_dict()
        #common parameters
        self.results_dir = self.params_dict['common_results_dir']
        self.column_size = self.params_dict['common_column_size']
        self.row_size = self.params_dict['common_row_size']
        self.key_ints = self.params_dict['common_key_ints']
        self.bad_pixels = self.params_dict['common_bad_pixels']

        #gain parameters
        self.nreps = self.params_dict['filter_nreps']
        self.nframes = self.params_dict['filter_nframes']
        self.min_signals = self.params_dict['gain_min_signals']

        self._logger.info(f'Parameters loaded:')
        self.params.print_contents()
        
        self._logger.info('Checking parameters in filter directory')
        #look for a json file in the filter directory
        if (not self.params.same_common_params(filter_dir)) \
            or (not self.params.same_offnoi_params(filter_dir) \
            or (not self.params.same_filter_params(filter_dir))):
            self._logger.error('Parameters in filter directory do not match')
            return
        try:
            self.event_map = af.get_array_from_file(
                filter_dir, 'event_map.npy')
            #set the directory where the filter data is stored
            self.filter_dir = filter_dir
            #this is the parent directory. data from this step is stored there
            self.common_dir = os.path.dirname(filter_dir)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.step_dir = os.path.join(
                self.common_dir, timestamp + f'_gain_{self.min_signals}_minsignals'
            )
        except:
            self._logger.error('Error loading filter data\n')
            return
        self._logger.info('Filter data loaded\n')

    def calculate(self) -> None:
        #create the working directory for the gain step
        self.step_dir = os.path.join(self.common_dir, 
                                     f'gain_{self.min_signals}_min_signals')
        os.makedirs(self.step_dir, exist_ok=True)
        self._logger.info(f'Created directory for gain step: {self.step_dir}')
        # and save the parameter file there
        self.params.save(os.path.join(self.step_dir, 'parameters.json'))
        
        fits = an.get_gain_fit(self.event_map, self.row_size, self.column_size, self.min_signals)
        np.save(os.path.join(self.step_dir, 'fit_mean.npy'), fits[0])
        np.save(os.path.join(self.step_dir, 'fit_sigma.npy'), fits[1])
        np.save(os.path.join(self.step_dir, 'fit_mean_error.npy'), fits[2])
        np.save(os.path.join(self.step_dir, 'fit_sigma_error.npy'), fits[3])