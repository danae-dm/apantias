"""Init:
Defines what modules are exposed to the user.
"""

import importlib.metadata
from . import config
from . import orchestrator

__version__ = importlib.metadata.version("apantias")
__author__ = "Florian Heinrich"
__credits__ = "HEPHY Vienna"

print(f"test APANTIAS version {__version__} loaded.")
