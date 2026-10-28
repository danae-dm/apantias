# Standard analysis orchestrates Dask/Zarr calls; both libraries ship incomplete
# stubs, so strict-mode unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
<<<<<<< HEAD

"""
Definition of the standard data analysis pipeline.

This module is the single place where the pipeline is defined: its stages,
their order, and which arrays count as final outputs (``AnalysisArrays``).
Infrastructure lives in dedicated modules and is only wired together here:
path layout in ``io.paths``, branch-point materialization in
``compute.materialize``, archiving in ``archive``.
"""
=======
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f

import logging
from dataclasses import dataclass
from pathlib import Path
<<<<<<< HEAD
from typing import Any, cast
=======
from typing import Self, cast
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f

import dask.array as da
import zarr

from apantias.archive import archive, cleanup_temp
from apantias.cluster import init
from apantias.compute import stats
from apantias.compute.materialize import Materializer
from apantias.compute.reductions import apply_pixelwise
from apantias.io.hdf5 import load_h5_dataset, split_h5_dataset_path
from apantias.io.paths import AnalysisPaths
from apantias.io.zarr import h5_to_zarr
from apantias.preflight_check import PreflightValidator
<<<<<<< HEAD
from apantias.settings import resolve_config
from apantias.transform.rechunk import rechunk_to_pixels
=======
from apantias.settings import AnalysisSettings, AppSettings, load_config

from .compute import stats
from .compute.reductions import apply_pixelwise
from .io.hdf5 import write_analysis_outputs_to_h5
from .io.zarr import h5_to_zarr
from .transform.rechunk import rechunk_to_pixels
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f

_logger = logging.getLogger(__name__)


<<<<<<< HEAD
=======
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


>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f
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

<<<<<<< HEAD
    def final_outputs(self) -> list[tuple[str, da.Array]]:
        """(name, array) pairs of the pipeline's final output stages."""
        return [
            ("offset", self.offset),
            ("common_modes", self.common_modes),
            ("median", self.median),
            ("slopes", self.slopes),
            ("msd", self.msd),
            ("signals_mean", self.signals_mean),
            ("pixelwise", self.pixelwise),
        ]

=======
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f

class StandardAnalysis:
    """
    Performs standard statistical analysis on zarr arrays.

<<<<<<< HEAD
    Data shape is (n_frames, n_rows, n_reps, n_cols).
    """

    def __init__(self, path: Path | str | None = None) -> None:
=======
    Data shape is (n_frames, n_cols, n_reps, n_rows).
    """

    def __init__(self, path: Path | str | AppSettings | None = None) -> None:
        if path is None:
            raw_config = load_config()
        elif isinstance(path, AppSettings):
            raw_config = path
        else:
            raw_config = load_config(Path(path))

>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f
        # Fail fast: validate inputs and infer FrameSettings before Dask starts
        self.config = PreflightValidator(resolve_config(path)).validate()

        # Dask cluster starts only after inputs are proven valid
        self._client, self._cluster = init(
            self.config.runtime.dask_temp, self.config.runtime.cpus, self.config.runtime.ram_mb
        )

        self.paths = AnalysisPaths.from_settings(self.config.analysis)
<<<<<<< HEAD
        self._materializer = Materializer(self._client, self.config.runtime.ram_mb)
=======
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f
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
            # The final outputs are already stored to zarr by _save_outputs;
            # archive reads them back from disk instead of recomputing.
            outputs = [(name, self.paths.path_for(name)) for name, _ in self.arrays.final_outputs()]
            archive(self.config, outputs)
        finally:
            client.close()
            cluster.close()
        cleanup_temp(self.config)

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
<<<<<<< HEAD
        self.arrays = arrays

    def _load_inputs(self) -> tuple[da.Array, da.Array]:
        # da.from_zarr sees the inner Zarr chunks (single-row) because the
        # frame-chunked store uses sharding. Downstream steps need full rows,
        # so rechunk axis 1 (rows) to n_rows here once.
        data_f = cast(
            da.Array,
            da.from_zarr(self.paths.raw_framewise).rechunk({1: self.config.frame.n_rows}),
=======

        _logger.info("Writing HDF5 archive.")
        write_analysis_outputs_to_h5(self.paths, self.config.analysis.h5_archive)

    def _load_inputs(self) -> tuple[da.Array, da.Array]:
        # da.from_zarr sees the inner Zarr chunks (single-column) because the
        # frame-chunked store uses sharding. Downstream steps need full columns,
        # so rechunk axis 1 (columns) to n_cols here once.
        data_f = cast(
            da.Array,
            da.from_zarr(self.paths.raw_framewise).rechunk({1: self.config.frame.n_cols}),
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f
        )
        data_p = cast(da.Array, da.from_zarr(self.paths.raw_pixelwise))
        return data_f, data_p

<<<<<<< HEAD
    def _load_offset(self, median: da.Array) -> da.Array:
        """Return the offset map to subtract.

        The external offset map if configured, otherwise the "own" offset:
        the per-pixel median across all frames computed in the previous step.
        """
        if self.ext_offset is None:
            return median
        h5_ref = split_h5_dataset_path(self.ext_offset)
        if h5_ref is not None:
            return load_h5_dataset(*h5_ref)
        # Split "store.zarr/group/array" so nested array paths resolve correctly;
        # a bare store.zarr whose root is an array loads with component=None.
        path_str = str(self.ext_offset)
        if ".zarr" in path_str:
            store_end = path_str.find(".zarr") + len(".zarr")
            component = path_str[store_end:].lstrip("/")
            return cast(da.Array, da.from_zarr(path_str[:store_end], component=component or None))
=======
    def _load_offset(self) -> da.Array:
        if self.ext_offset is None:
            # Shape (n_cols, n_rows) broadcasts into the 4D frame array.
            return da.zeros((self.config.frame.n_cols, self.config.frame.n_rows), chunks=-1)
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f
        return cast(da.Array, da.from_zarr(self.ext_offset))

    def _build_pipeline(self, data_f: da.Array, data_p: da.Array) -> AnalysisArrays:
        _logger.info("Start calculating median.")
        median = stats.compute_median(data_p)

        _logger.info("Start applying offset.")
<<<<<<< HEAD
        offset = self._load_offset(median)
        offset_corr = stats.compute_offset_corr(data_f, offset)
        # Branch point: used by common_modes and subtract.
        offset_corr = self._materializer.persist_or_store(offset_corr, self.paths.offset_corr)

        _logger.info("Start common mode correction.")
        common_modes = stats.compute_common_modes(offset_corr)
        signals = stats.subtract(offset_corr, common_modes)
        # Branch point: used by slopes, msd and signals_mean.
        signals = self._materializer.persist_or_store(signals, self.paths.signals)

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

    def _save_outputs(self, arrays: AnalysisArrays) -> None:
        _logger.info("Writing final outputs.")
        outputs = arrays.final_outputs()
        # da.store's batch form needs real array targets that support
        # ``out[index] = x``; plain paths only work with single-array
        # ``to_zarr``, so open a zarr array for each output first.
        targets = [self._open_zarr_output(self.paths.path_for(name), arr) for name, arr in outputs]
        da.store([arr for _, arr in outputs], targets, lock=False)
        _logger.info("Done.")

    @staticmethod
    def _open_zarr_output(path: Path, arr: da.Array) -> Any:
        """Create (overwriting) the zarr array at *path* with chunks matching *arr*."""
        return zarr.open_array(
            str(path),
            mode="w",
            shape=arr.shape,
            chunks=arr.chunksize,
            dtype=arr.dtype,
            zarr_format=3,
        )
=======
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
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f
