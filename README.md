new features/bugs/requests:
https://github.com/shakamaran/apantias/issues

Steps to upload package:
https://packaging.python.org/en/latest/tutorials/packaging-projects/

Uploading to PyPi works automatically when merging to the github main-branch.
Install it via pip in your conda environemnt. It is currently not uploaded to conda-forge, so
it cannot be found in conda.

Be sure to update the version both in __init__.py and pyproject.toml before merging to main!

COL vs ROW Convention:
In ROOT its (col, row), but:
data is represented as (frame,row,nreps,col), so i will use (row, col) here, since its the
natural order in the array.

How-To get the package in JupyterHub:

first make sure anaconda is loaded as a module on clip. (documentation of modules: https://docs.vbc.ac.at/books/scientific-computing/page/module-collection)
MAKE SURE THERE IS NO OTHER PYTHON LOADED AS A MODULE

in the terminal on clip:

# Create the environment with Python 3.8 and Jupyter in a specific folder. its recommended to install it in you personal folder on CLIP.
conda create --prefix /users/user.name/.conda/envs/myenv python=3.8 jupyter

# Activate the environment
conda activate /path/to/envs/myenv

# install apantias package
pip install apantias

# Add the environment to JupyterHubs Conda Manager
python -m ipykernel install --user --name myenv --display-name "myname"

the apantias package already uses a lot of common packages (matplotlib, numpy,..). If you need more packages just install them in the conda manager on jupyterhub.

# Note, that the apantias package cannot be updated in conda directly! Use
pip install --upgrade apantias

For Development the package can be loaded in conda in editable mode:

# Activate the Conda environment
conda activate myenv

# Navigate to the package directory
cd /path/to/your/package

# Install the package in editable mode
pip install -e .