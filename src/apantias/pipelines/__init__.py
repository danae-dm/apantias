"""Analysis pipelines.

Each submodule implements a distinct analysis pipeline (e.g. the standard
DANAE reduction). New analysis types should live here as sibling modules.
"""

from .standard import StandardPipeline

__all__ = ["StandardPipeline"]
