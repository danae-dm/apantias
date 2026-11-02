"""Analysis pipelines.

Each submodule implements a distinct analysis pipeline (e.g. the standard
DANAE reduction). New analysis types should live here as sibling modules.
"""

from .standard import StandardAnalysis

__all__ = ["StandardAnalysis"]
