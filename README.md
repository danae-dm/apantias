import nproan.analysis
import nproan.analysis_funcs
import nproan.display
import nproan.fitting
import nproan.pyroot_funcs
import nproan.roan_steps

...these are imported when loading the package.

Modules:

roan_steps:
Contains classes for OffNoi, Filter and Gain step. Each class is initialized
with a path to a parameter file in .json format.
The .calculate() method performs all the computations in the step. It also
handles saving of files (paths are defined in the parameter file)
This is exposed to the user when importing nproan.

analysis:
Contains functions used for analysing/manipulating data. More complex
functions are in here (eg more complex than np.nanmean).
Funtions used in roan_steps should be from this module.
This is exposed to the user when importing nproan.

display:
Contains functions used for displaying data.
This is exposed to the user when importing nproan.

analysis_funcs:
Contains functions used for analysing/manipulating data. They
are also used in the analysis module.
This is exposed to the user when importing nproan.

fitting:
Contains functions for fitting to data. They are used in the analysis
step.
This is NOT exposed to the user when importing nproan. It can be imported
however (from nproan import fitting)

logger:
Class that provides logging capabilities.
This is NOT exposed to the user when importing nproan. It can be imported
however (from nproan import logger). But it would make no sense.

parallel_funcs:
Contains parallelized versions of simple numpy functions. The are used in
analysis/analysis_funcs

params:
Contains functions for handling parameter .json files.
step.
This is NOT exposed to the user when importing nproan. It can be imported
however (from nproan import params) It would make no sense, since its only
used by roan_steps.

pyroot_funcs:
Contains functions to interface with ROOT using the pyroot package.
This is NOT exposed to the user when importing nproan. It takes a long time to
load and should be only loaded when neccesary.

TODO:
- use dictionaries in .npy files
- add additional info to .npz file output
- bad slopes should be ignored in the fitting in the offnoi step
- add infos to plots
- check documentation of functions
- wiki schreiben
- add type hints and write nice documentation (almost done)
- consider combining analysis and analysis_funcs


COL vs ROW Convention:

In ROOT its (col, row), but:
data is represented as (frame,row,nreps,col), so i will use (row, col) here
Size if data is (frames,column_size,nreps, row_size)


My current philosophy for doing stuff:
- Parameters from the parameter file are stored in the class variables
- Parameters can be boolean or int. for thresholds 0 means no calculation is performed.
- parameter file is in JSON format
- for every step class, the whole parameter file must be loaded
- the filter class loads the offnoi data, and checks consistency with the
    parameter file
- the gain class loads the filter data, and checks consistency with the
    parameter file
- one directory (the "common_dir") should be created in a "results" directory 
    on \scratch this should only be done once in the offnoi step
    in there, a directory "offnoi" should be created
    the parameter file must be placed there after calculation
- in the filter step a folder path to the "offnoi" directory should be
    provided along with the parameters
    consistency of the parameters (from last steps) will be checked
- in the gain step a folder path to the "filter" directory should be
    provided along with the parameters
    consistency of the parameters (from last steps) will be checked
- functions that return values should be named get_something(), its preferred To
    return a value. 
- provide some functions that do data manipulation inplace
- data that is loaded from file in the filter or gain step is stored in a class 
    variable

Steps to upload package:
https://packaging.python.org/en/latest/tutorials/packaging-projects/