"""Init:
Defines what modules are exposed to the user.
"""

import importlib.metadata

__version__ = importlib.metadata.version("apantias")
__author__ = "Florian Heinrich"
__credits__ = "HEPHY Vienna"

from . import analysis
from . import display
from . import utils
from . import file_io
from . import bin_to_h5
from .standard import Default

print(f"APANTIAS version {__version__} loaded.")
