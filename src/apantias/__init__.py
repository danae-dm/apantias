"""Init:
Defines what modules are exposed to the user.
"""

import importlib.metadata
import logging
import sys

from apantias import utils
from apantias import client
from apantias import display
from apantias import standard
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
__all__ = ["utils", "client", "display", "standard"]


# multiprocessing.current_process().name is 'MainProcess' in the parent Jupyter kernel.
# Dask's LocalCluster spawns workers via multiprocessing, where the name becomes
# something like 'ForkProcess-1', 'SpawnPoolWorker-2', etc.
if multiprocessing.current_process().name == "MainProcess":
    print(f"APANTIAS version {__version__} loaded.")
