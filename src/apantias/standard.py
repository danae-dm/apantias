# Standard analysis orchestrates Dask/Zarr calls; both libraries ship incomplete
# stubs, so strict-mode unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

"""
Definition of the standard data analysis pipeline.

This module is the single place where the pipeline is defined: its stages,
their order, and which arrays count as final outputs (``AnalysisArrays``).
Infrastructure lives in dedicated modules and is only wired together here:
path layout in ``io.paths``, branch-point materialization in
``compute.materialize``, archiving in ``archive``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
from apantias.settings import resolve_config
from apantias.transform.rechunk import rechunk_to_pixels

_logger = logging.getLogger(__name__)


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


class StandardAnalysis:
    """
    Performs standard statistical analysis on zarr arrays.

    Data shape is (n_frames, n_rows, n_reps, n_cols).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        # Fail fast: validate inputs and infer FrameSettings before Dask starts
        self.config = PreflightValidator(resolve_config(path)).validate()

        # Dask cluster starts only after inputs are proven valid
        self._client, self._cluster = init(
            self.config.runtime.dask_temp, self.config.runtime.cpus, self.config.runtime.ram_mb
        )

        self.paths = AnalysisPaths.from_settings(self.config.analysis)
        self._materializer = Materializer(self._client, self.config.runtime.ram_mb)
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
        self.arrays = arrays

    def _load_inputs(self) -> tuple[da.Array, da.Array]:
        # da.from_zarr sees the inner Zarr chunks (single-row) because the
        # frame-chunked store uses sharding. Downstream steps need full rows,
        # so rechunk axis 1 (rows) to n_rows here once.
        data_f = cast(
            da.Array,
            da.from_zarr(self.paths.raw_framewise).rechunk({1: self.config.frame.n_rows}),
        )
        data_p = cast(da.Array, da.from_zarr(self.paths.raw_pixelwise))
        return data_f, data_p

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
        return cast(da.Array, da.from_zarr(self.ext_offset))

    def _build_pipeline(self, data_f: da.Array, data_p: da.Array) -> AnalysisArrays:
        _logger.info("Start calculating median.")
        median = stats.compute_median(data_p)

        _logger.info("Start applying offset.")
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
