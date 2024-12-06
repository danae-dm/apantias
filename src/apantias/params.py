import os
import fnmatch
import json

from . import logger


class Params:

    _logger = logger.Logger(__name__, "info").get_logger()

    """
    To change/add parameters, edit/add them here.
    1) Add the parameter to the common_params, offnoi_params, filter_params, or gain_params dictionaries.
    2) Add the type of the parameter to the params_types dictionary.
    3) Add the parameter to the required_params list if it has no default value and is required.
    Also change the load() function in the RoanSteps class.
    """
    common_params = {
        "common_results_dir": "",  # str
        "common_available_cpus": 16,  # int
        "common_mem_per_cpu_gb": 2,  # int
        "common_attributes": {},  # dict
    }
    offnoi_params = {
        "offnoi_data_file": "",  # list of str
        "offnoi_nframes_eval": [0, -1, 1],  # list of ints
        "offnoi_nreps_eval": [0, -1, 1],  # list of ints
        "offnoi_comm_mode": True,  # bool
        "offnoi_thres_bad_slopes": 3,  # float
    }
    filter_params = {
        "filter_data_file": "",  # list of str
        "filter_nframes_eval": [0, -1, 1],  # list of ints
        "filter_nreps_eval": [0, -1, 1],  # list of ints
        "filter_comm_mode": True,  # bool
        "filter_thres_event_prim": 3,  # float
        "filter_thres_event_sec": 3,  # float
        "filter_thres_bad_slopes": 3,  # float
    }
    gain_params = {"gain_dummy": 5}  # int

    # types are checked when they are read
    params_types = {
        "common_results_dir": str,
        "common_available_cpus": int,
        "common_mem_per_cpu_gb": int,
        "common_attributes": dict,
        "offnoi_data_file": str,
        "offnoi_nframes_eval": list,
        "offnoi_nreps_eval": list,
        "offnoi_comm_mode": bool,
        "offnoi_thres_bad_slopes": (int, float),
        "filter_data_file": str,
        "filter_nframes_eval": list,
        "filter_nreps_eval": list,
        "filter_comm_mode": bool,
        "filter_thres_event_prim": (int, float),
        "filter_thres_event_sec": (int, float),
        "filter_thres_bad_slopes": (int, float),
        "gain_dummy": int,
    }

    # required parameters, where there is no default value
    # file cannot be loaded if these are missing
    required_params = [
        "common_results_dir",
        "offnoi_data_file",
        "filter_data_file",
        "common_available_cpus",
        "common_mem_per_cpu_gb",
    ]

    def __init__(self, json_path: str = None):
        self.default_dict = {
            **self.common_params,
            **self.offnoi_params,
            **self.filter_params,
            **self.gain_params,
        }
        self.inp_dict = None
        self.param_dict = None
        if json_path is not None:
            self.update(json_path)
            self.check_types()
        else:
            self._logger.error("No parameter file provided.")
            self._logger.error("Created default parameter file.")
            self.save_default_file()
            self._logger.error("Add all required parameters to default.")
            self._logger.error("Add the path to the parameter file as an argument.")
            raise ValueError("No parameter file provided.")

    def update(self, json_path: str) -> None:
        try:
            with open(json_path) as f:
                self.inp_dict = json.load(f)
        except:
            self._logger.error("Error loading the parameter file.")
            self.save_default_file()
            self._logger.error(
                "A default parameter file has been saved to the current directory."
            )
            self.param_dict = None
            return
        self.param_dict = self.default_dict.copy()
        # check consistency of the input dict with the default dict
        for key, value in self.inp_dict.items():
            if key not in self.default_dict:
                self.save_default_file()
                self._logger.error(
                    "A default parameter file has been saved to the current directory."
                )
                raise KeyError(f"{key} is not a valid parameter.")
            else:
                self.param_dict[key] = value
        # check for missing parameters, using default if not required
        # if parameter has no default, set param_dict to None
        for key, value in self.param_dict.items():
            if value is None:
                if key in self.required_params:
                    self._logger.error(f"{key} is missing in the file.")
                    self._logger.error("Please provide a complete parameter file")
                    self.save_default_file()
                    self._logger.error(
                        "A default parameter file has been saved to the current directory."
                    )
                    self.param_dict = None
                    break
                else:
                    self._logger.error(
                        f"{key} is missing. Using default: {self.default_dict[key]}"
                    )

    def check_types(self) -> None:
        for key, value in self.param_dict.items():
            if key not in self.params_types:
                raise TypeError(f"There is no type defined for {key}.")
            else:
                expected_type = self.params_types[key]
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected {key} to be of type {expected_type}.")

    def get_dict(self) -> dict:
        return self.param_dict

    def print_contents(self) -> None:
        for key, value in self.param_dict.items():
            self._logger.info(f"{key}: {value}")

    def info(self) -> None:
        self._logger.info("The following parameters must be provided:")
        self._logger.info("--common parameters:")
        for key in self.common_params.keys():
            self._logger.info(key)
        self._logger.info("--offnoi parameters:")
        for key in self.offnoi_params.keys():
            self._logger.info(key)
        self._logger.info("--filter parameters:")
        for key in self.filter_params.keys():
            self._logger.info(key)
        self._logger.info("--gain parameters:")
        for key in self.gain_params.keys():
            self._logger.info(key)
        self._logger.info("required parameters:")
        for key in self.required_params:
            self._logger.info(key)

    def save_default_file(self, path: str = None) -> None:
        # if no path is provided, save to the current directory
        self._logger.info(f"path: {path}")
        if path is None:
            path = os.path.join(os.getcwd(), "default_params.json")
        else:
            path = os.path.join(path, "default_params.json")
        with open(path, "w") as f:
            json.dump(self.default_dict, f, indent=4)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.param_dict, f, indent=4)

    def get_json_file_name_in_folder(self, folder_path: str) -> str:
        count = 0
        json_file = ""
        for file in os.listdir(folder_path):
            if fnmatch.fnmatch(file, "*.json"):
                json_file = os.path.join(folder_path, file)
                count += 1
        if count == 1:
            return json_file
        else:
            return None
