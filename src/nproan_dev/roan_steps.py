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

    def __init__(self, prm_file: str = None) -> None:
        self.load(prm_file)
        self._logger.info('PreprocessData object created')

    def load(self, prm_file: str) -> None:
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

        #directories
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = os.path.basename(self.filter_bin_file)[:-4]
        self.common_dir = os.path.join(
            self.results_dir, timestamp + '_' + filename)
        self.step_dir = os.path.join(self.common_dir, 'preprocess')
        self._logger.info(f'Parameters loaded:')
        self.params.print_contents()

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
        virtual_memory = psutil.virtual_memory()
        available_memory_in_bytes = virtual_memory.available
        available_memory_in_gb = available_memory_in_bytes / (1024 ** 3)

        self._logger.info(f'RAM available: {available_memory_in_gb:.1f} GB')
        self._logger.info(f'Estimated RAM usage: {estimated_ram_usage:.1f} GB')
        steps_needed = int(estimated_ram_usage / available_memory_in_gb) + 1
        self._logger.info(f'Steps needed: {steps_needed}')

        #is set to 8, this is the offset (header) in the .bin file
        offset = 8

        #calculate how much frames can be loaded at once:
        frames_per_step = int(self.offnoi_nframes / steps_needed)
        for step in range(steps_needed):
            self._logger.info(f'Performing step {step+1} of {steps_needed} total Steps')
            data, offset = an.get_data_2(self.offnoi_bin_file, self.column_size, self.row_size, self.key_ints, self.offnoi_nreps, frames_per_step, offset)
            
            #exclude nreps_eval from data
            if self.nreps_eval:
                data = an.exclude_nreps_eval(data, self.nreps_eval)
                self._logger.debug(f'Shape of data: {data.shape}')
            #set values of all frames and nreps of bad pixels to nan
            if self.bad_pixels:
                data = an.set_bad_pixellist_to_nan(data, self.bad_pixels)
            #delete bad frames from data
            if self.thres_bad_frames != 0 or self.thres_mips != 0:
                data = an.exclude_mips_and_bad_frames(data, self.offnoi_thres_mips, 
                                                      self.offnoi_thres_bad_frames)
                self._logger.debug(f'Shape of data: {data.shape}')

            '''
            TODO here:
            - calculate offset_raw on the raw data and save/update it
            - calculate data - offset_raw
            - correct common mode
            - calculate rndr signals and save/update it
            - calculate bad slopes and save/update them

            - do the same preprocessing for the filter data

            - rewrite offnoi and filter classes to use the new data
            '''



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

        data = an.get_data(self.bin_file, self.column_size, self.row_size, self.key_ints, self.nreps, self.nframes)
        gc.collect()
        #delete nreps_eval from data
        if self.nreps_eval:
            data = an.exclude_nreps_eval(data, self.nreps_eval)
            self._logger.debug(f'Shape of data: {data.shape}')
        #set values of all frames and nreps of bad pixels to nan
        if self.bad_pixels:
            data = an.set_bad_pixellist_to_nan(data, self.bad_pixels)
        #deletes bad frames from data
        if self.thres_bad_frames != 0:
            data = an.exclude_bad_frames(data, self.thres_bad_frames, self.step_dir)
            self._logger.debug(f'Shape of data: {data.shape}')
        #deletes frames with mips above threshold from data
        if self.thres_mips != 0:
            data = an.exclude_mips_frames(data, self.thres_mips)
            self._logger.debug(f'Shape of data: {data.shape}')
        #calculate offset_raw on the raw data and save it
        avg_over_frames = af.get_avg_over_frames(data)
        np.save(os.path.join(self.step_dir, 'offset_raw.npy'),
                avg_over_frames)
        #calculate offset and save it
        avg_over_frames_and_nreps = af.get_avg_over_frames_and_nreps(data)
        np.save(os.path.join(self.step_dir, 'offset.npy'),
                avg_over_frames_and_nreps)
        #offset the data and correct for common mode if necessary
        data -= avg_over_frames[np.newaxis,:,:,:]
        del avg_over_frames
        gc.collect()
        if self.comm_mode is True:
            an.correct_common_mode(data)
        #calculate rndr signals and save it
        avg_over_nreps = af.get_avg_over_nreps(data)
        np.save(os.path.join(self.step_dir, 'rndr_signals.npy'),
                avg_over_nreps)
        #calculate fitted offset and noise and save it (including fit errors)
        fit_unbinned = fit.get_unbinned_fit_gauss(avg_over_nreps)
        fit_curve_fit = fit.get_fit_gauss(avg_over_nreps)
        np.save(os.path.join(self.step_dir, 'offnoi_fit_unbinned.npy'), fit_unbinned)
        np.save(os.path.join(self.step_dir, 'offnoi_fit.npy'), fit_curve_fit)
        if self.thres_bad_slopes != 0:
            bad_slopes_pos, bad_slopes_data, bad_slopes_value = an.get_bad_slopes(data, self.thres_bad_slopes, self.step_dir)
            np.save(os.path.join(self.step_dir, 'bad_slopes_pos.npy'), bad_slopes_pos)
            np.save(os.path.join(self.step_dir, 'bad_slopes_data.npy'), bad_slopes_data)
            np.save(os.path.join(self.step_dir, 'bad_slopes_value.npy'), bad_slopes_value)

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
            #offset_raw is quite big. deleted after use
            self.offset_raw = af.get_array_from_file(
                offnoi_dir, 'offset_raw.npy')
            self._logger.debug(self.offset_raw.shape)
            if self.offset_raw is None:
                self._logger.error('Error loading offset_raw data\n')
                return
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

        data = an.get_data(self.bin_file, self.column_size, self.row_size, self.key_ints, self.nreps, self.nframes)
        gc.collect()
        if self.nreps_eval:
            data = an.exclude_nreps_eval(data, self.nreps_eval)
            #number of nreps in offnoi can be different from filter
            #BUT: #offnoi must be bigger than #filter!
            self._logger.debug(f'Shape of offset_raw: {self.offset_raw.shape}')
            if self.offset_raw.shape[1] < data.shape[2]:
                self.logger.error('Number of nreps in offnoi is smaller than in filter. Exiting.')
                raise ValueError('Number of nreps in offnoi is smaller than in filter.')
            elif self.offset_raw.shape[1] != data.shape[2]:
                self.offset_raw = an.exclude_nreps_eval_offset_raw(self.offset_raw, self.nreps_eval)
            self._logger.debug(f'Shape of data: {data.shape}')
            self._logger.debug(f'Shape of offset_raw: {self.offset_raw.shape}')
        #omit bad pixels and mips frames
        if self.bad_pixels:
            data = an.set_bad_pixellist_to_nan(data, self.bad_pixels)
        if self.thres_bad_frames != 0:
            data = an.exclude_bad_frames(data, self.thres_bad_frames, self.step_dir)
            self._logger.debug(f'Shape of data: {data.shape}')
        if self.thres_mips != 0:
            data = an.exclude_mips_frames(data, self.thres_mips)
            self._logger.debug(f'Shape of data: {data.shape}')
        #offset the data and correct for common mode if necessary
        data = data - self.offset_raw[np.newaxis,:,:,:]
        self.offset_raw = None
        gc.collect()
        if self.comm_mode:
            an.correct_common_mode(data)
        if self.use_fitted_offset:
            #take care here, offset fitted can contain np.nan
            data -= np.nan_to_num(self.offset_fitted[np.newaxis,:,np.newaxis,:])
        avg_over_nreps = af.get_avg_over_nreps(data)
        np.save(os.path.join(self.step_dir, 'rndr_signals.npy'),
                avg_over_nreps)
        #calculate event map and save it
        event_map = an.calc_event_map(avg_over_nreps, self.noise_fitted, self.thres_event)
        np.save(os.path.join(self.step_dir, 'event_map.npy'),
                event_map)
        np.save(os.path.join(self.step_dir, 'sum_of_event_signals.npy'),
                an.get_sum_of_event_signals(event_map, self.row_size, self.column_size))
        np.save(os.path.join(self.step_dir, 'sum_of_event_counts.npy'),
                an.get_sum_of_event_counts(event_map, self.row_size, self.column_size))
        if self.thres_bad_slopes != 0:
            bad_slopes_pos, bad_slopes_data, bad_slopes_value = an.get_bad_slopes(data, self.thres_bad_slopes, self.step_dir)
            np.save(os.path.join(self.step_dir, 'bad_slopes_pos.npy'), bad_slopes_pos)
            np.save(os.path.join(self.step_dir, 'bad_slopes_data.npy'), bad_slopes_data)
            np.save(os.path.join(self.step_dir, 'bad_slopes_value.npy'), bad_slopes_value)

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