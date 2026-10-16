"""On-disk layout of the Zarr stores used by the standard analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

from apantias.settings import AnalysisSettings


@dataclass(frozen=True)
class AnalysisPaths:
    """All on-disk locations used by the standard analysis."""

    raw_framewise: Path
    raw_pixelwise: Path
    median: Path
    offset: Path
    offset_corr: Path
    common_modes: Path
    signals: Path
    slopes: Path
    msd: Path
    signals_mean: Path
    pixelwise: Path
    data_root: Path
    temp_root: Path

    @classmethod
    def from_settings(cls, analysis: AnalysisSettings) -> Self:
        zarr_data = Path(analysis.zarr_data)
        zarr_temp = Path(analysis.zarr_temp)
        return cls(
            raw_framewise=zarr_data / "frame_chunked",
            raw_pixelwise=zarr_data / "pixel_chunked",
            median=zarr_temp / "median",
            offset=zarr_temp / "offset",
            offset_corr=zarr_temp / "offset_corr",
            common_modes=zarr_temp / "common_modes",
            signals=zarr_temp / "signals",
            slopes=zarr_temp / "slopes",
            msd=zarr_temp / "msd",
            signals_mean=zarr_temp / "signals_mean",
            pixelwise=zarr_temp / "pixelwise",
            data_root=zarr_data,
            temp_root=zarr_temp,
        )

    def path_for(self, name: str) -> Path:
        """Path of a final output stage by its array name (e.g. ``"slopes"``).

        Works because every final output stage has a field of the same name.
        """
        return cast(Path, getattr(self, name))
