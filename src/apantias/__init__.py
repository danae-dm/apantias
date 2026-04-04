"""Init:
Defines what modules are exposed to the user.
"""

import importlib.metadata

__version__ = importlib.metadata.version("apantias")
__author__ = "Florian Heinrich"
__credits__ = "HEPHY Vienna"

print(f"APANTIAS version {__version__} loaded.")
