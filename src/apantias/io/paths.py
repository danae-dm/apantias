"""On-disk layout of the raw input Zarr stores used by the standard analysis."""

from dataclasses import dataclass
from pathlib import Path

from apantias.settings import AnalysisSettings


@dataclass(frozen=True)
class AnalysisPaths:
    """Locations of the raw input stores and the two Zarr roots.

    This class deliberately does *not* list the analysis outputs (median,
    signals, ...): the ``Pipeline`` registry in ``io.outputs`` is the single
    source of truth for where products are stored, resolving each stage to
    ``<zarr_temp>/<stage name>`` (or ``<zarr_data>/<stage name>`` for raw
    data locations).  Keeping only the inputs here avoids two drifting
    definitions of the same layout.
    """

    raw_framewise: Path
    raw_pixelwise: Path
    data_root: Path
    temp_root: Path

    @classmethod
    def from_settings(cls, analysis: AnalysisSettings) -> "AnalysisPaths":
        zarr_data = Path(analysis.zarr_data)
        zarr_temp = Path(analysis.zarr_temp)
        return cls(
            raw_framewise=zarr_data / "frame_chunked",
            raw_pixelwise=zarr_data / "pixel_chunked",
            data_root=zarr_data,
            temp_root=zarr_temp,
        )
