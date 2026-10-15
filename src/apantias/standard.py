# Standard analysis orchestrates Dask/Zarr calls; both libraries ship incomplete
# stubs, so strict-mode unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import dask.array as da
import zarr

from apantias.cluster import init
from apantias.preflight_check import PreflightValidator
from apantias.settings import AnalysisSettings, AppSettings, load_config

from .compute import stats
from .compute.reductions import apply_pixelwise
from .io.hdf5 import write_analysis_outputs_to_h5
from .io.zarr import h5_to_zarr
from .transform.rechunk import rechunk_to_pixels

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisPaths:
    """All on-disk locations used by the standard analysis."""

    raw_framewise: Path
    raw_pixelwise: Path
    median: Path
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


@dataclass
class AnalysisArrays:
    """Lazy (or materialized) arrays at each pipeline stage."""

    raw_framewise: da.Array
    raw_pixelwise: da.Array
    median: da.Array
    offset: da.Array
    offset_corr: da.Array
    common_modes: da.Array
    signals: da.Array
    slopes: da.Array
    msd: da.Array
    signals_mean: da.Array
    pixelwise: da.Array


class StandardAnalysis:
    """
    Performs standard statistical analysis on zarr arrays.

    Data shape is (n_frames, n_cols, n_reps, n_rows).
    """

    def __init__(self, path: Path | str | AppSettings | None = None) -> None:
        if path is None:
            raw_config = load_config()
        elif isinstance(path, AppSettings):
            raw_config = path
        else:
            raw_config = load_config(Path(path))

        # Fail fast: validate inputs and infer FrameSettings before Dask starts
        self.config = PreflightValidator(raw_config).validate()

        # Dask cluster starts only after inputs are proven valid
        self._client, self._cluster = init(self.config.runtime.dask_temp, self.config.runtime.cpus)

        self.paths = AnalysisPaths.from_settings(self.config.analysis)
        self.h5_path = Path(self.config.analysis.h5_file)
        self.slope_threshold = self.config.analysis.slope_threshold
        raw_ext = self.config.analysis.ext_offset
        if raw_ext is not None and str(raw_ext) != "None":
            self.ext_offset = Path(raw_ext)
        else:
            self.ext_offset = None

    def run(self) -> None:
        client, cluster = self._client, self._cluster
        try:
            self._run_analysis()
        finally:
            client.close()
            cluster.close()

    def _run_analysis(self) -> None:
        zarr.open_group(self.paths.temp_root, mode="a")
        zarr.open_group(self.paths.data_root, mode="a")

        _logger.info("Start writing bin to zarr store.")
        h5_to_zarr(self.h5_path, self.paths.raw_framewise, self.config.frame, self.config.analysis)
        rechunk_to_pixels(self.paths.raw_framewise, self.paths.raw_pixelwise)
        _logger.info("Done.")

        data_f, data_p = self._load_inputs()
        arrays = self._build_pipeline(data_f, data_p)
        self._save_outputs(arrays)

        _logger.info("Writing HDF5 archive.")
        write_analysis_outputs_to_h5(self.paths, self.config.analysis.h5_archive)

    def _load_inputs(self) -> tuple[da.Array, da.Array]:
        # da.from_zarr sees the inner Zarr chunks (single-column) because the
        # frame-chunked store uses sharding. Downstream steps need full columns,
        # so rechunk axis 1 (columns) to n_cols here once.
        data_f = cast(
            da.Array,
            da.from_zarr(self.paths.raw_framewise).rechunk({1: self.config.frame.n_cols}),
        )
        data_p = cast(da.Array, da.from_zarr(self.paths.raw_pixelwise))
        return data_f, data_p

    def _load_offset(self) -> da.Array:
        if self.ext_offset is None:
            # Shape (n_cols, n_rows) broadcasts into the 4D frame array.
            return da.zeros((self.config.frame.n_cols, self.config.frame.n_rows), chunks=-1)
        return cast(da.Array, da.from_zarr(self.ext_offset))

    def _build_pipeline(self, data_f: da.Array, data_p: da.Array) -> AnalysisArrays:
        _logger.info("Start calculating median.")
        median = stats.compute_median(data_p)

        _logger.info("Start applying offset.")
        offset = self._load_offset()
        offset_corr = stats.compute_offset_corr(data_f, offset, median)
        # Branch point: used by common_modes and subtract.
        offset_corr = self._persist_or_store(offset_corr, self.paths.offset_corr)

        _logger.info("Start common mode correction.")
        common_modes = stats.compute_common_modes(offset_corr)
        # Materialized too: it is a final output and must exist in Zarr for
        # the archive pass, not only as an intermediate for subtract.
        common_modes = self._persist_or_store(common_modes, self.paths.common_modes)
        signals = stats.subtract(offset_corr, common_modes)
        # Branch point: used by slopes, msd and signals_mean.
        signals = self._persist_or_store(signals, self.paths.signals)

        _logger.info("Start calculating slopes, msd and mean signals.")
        slopes = stats.compute_slopes(signals)
        msd = stats.compute_msd(signals)
        signals_mean = stats.compute_signals_mean(signals)

        _logger.info("Start pixelwise calculation.")
        pixelwise = apply_pixelwise(data_p, lambda t: t.max() - t.min())

        return AnalysisArrays(
            raw_framewise=data_f,
            raw_pixelwise=data_p,
            median=median,
            offset=offset,
            offset_corr=offset_corr,
            common_modes=common_modes,
            signals=signals,
            slopes=slopes,
            msd=msd,
            signals_mean=signals_mean,
            pixelwise=pixelwise,
        )

    def _persist_or_store(self, arr: da.Array, path: Path) -> da.Array:
        """Materialize a branch-point array so it is not recomputed.

        Uses an in-memory persist when the array fits comfortably in the
        cluster RAM, otherwise writes to Zarr and reads it back.
        """
        threshold = self._in_memory_threshold()
        if 0 < arr.nbytes < threshold:
            _logger.info("Persist %s in memory (%.2f MB).", path.name, arr.nbytes / 1e6)
            persisted = cast(da.Array, self._client.persist(arr))
            # Also write the store: downstream stages read from memory, but the
            # HDF5 archive pass needs every stage materialized in Zarr.
            persisted.to_zarr(path, overwrite=True)
            return persisted

        _logger.info("Store %s to zarr (%.2f MB).", path.name, arr.nbytes / 1e6)
        arr.to_zarr(path, overwrite=True)
        return cast(da.Array, da.from_zarr(path))

    def _in_memory_threshold(self) -> int:
        # Use a quarter of the configured RAM as the per-array persist budget.
        ram_mb = self.config.runtime.ram_mb
        if ram_mb <= 0:
            return 0
        return int(ram_mb * 1024 * 1024 * 0.25)

    def _save_outputs(self, arrays: AnalysisArrays) -> None:
        _logger.info("Writing final outputs.")
        outputs = [
            arrays.median,
            arrays.slopes,
            arrays.msd,
            arrays.signals_mean,
            arrays.pixelwise,
        ]
        paths = [
            self.paths.median,
            self.paths.slopes,
            self.paths.msd,
            self.paths.signals_mean,
            self.paths.pixelwise,
        ]
        # da.store needs writable array targets; Path objects do not support
        # item assignment, so open one Zarr array per output first.
        targets = [
            zarr.open_array(
                str(path),
                mode="w-",
                shape=out.shape,
                chunks=out.chunksize,
                dtype=out.dtype,
                zarr_format=3,
            )
            for out, path in zip(outputs, paths, strict=True)
        ]
        da.store(outputs, targets, lock=False)
        _logger.info("Done.")
