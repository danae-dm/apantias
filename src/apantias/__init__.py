__version__ = "2.0.1"
__author__ = "Florian Heinrich"
__credits__ = "HEPHY Vienna"

from . import analysis
from . import display
from . import utils
from . import file_io
from . import bin_to_h5
from . import standard
from .standard import Default

print(f"APANTIAS version {__version__} loaded.")
