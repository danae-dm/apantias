"""Init:
Defines what modules are exposed to the user.
"""

import importlib.metadata
import logging
import sys


from . import core
from . import standard
from . import settings
import multiprocessing


# Set up logging for interactive environments (Jupyter)
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)  # ← explicitly use stdout
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s: %(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)

__version__ = importlib.metadata.version("apantias")
__author__ = "Florian Heinrich"
__credits__ = "HEPHY Vienna"
# controls what is imported by "from apantiuas import *"
__all__ = ["core", "settings", "standard"]


# multiprocessing.current_process().name is 'MainProcess' in the parent Jupyter kernel.
# Dask's LocalCluster spawns workers via multiprocessing, where the name becomes
# something like 'ForkProcess-1', 'SpawnPoolWorker-2', etc.
if multiprocessing.current_process().name == "MainProcess":
    print(f"APANTIAS version {__version__} loaded.")
