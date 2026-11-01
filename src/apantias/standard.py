# Standard analysis orchestrates Dask/Zarr calls; both libraries ship incomplete
# stubs, so strict-mode unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

"""Definition of the standard data analysis pipeline.

This module declares the analysis stages in a ``Pipeline``.  Each stage is
registered together with metadata that controls where it is stored and whether
it is a branch point.  How the stage is archived is decided centrally by the
HDF5 writer based on the configured chunk threshold.  Infrastructure lives in
dedicated modules and is only wired together here: path layout in ``io.paths``,
the pipeline registry in ``io.outputs``, branch-point materialization in
``_persist_or_store``, and archiving in ``io.hdf5``.
"""

import logging
from pathlib import Path
from typing import cast

import dask.array as da
import zarr

from apantias.cluster import init
from apantias.compute import stats
from apantias.compute.reductions import apply_pixelwise
from apantias.io.hdf5 import write_analysis_outputs_to_h5
from apantias.io.outputs import BRANCH, Pipeline
from apantias.io.paths import AnalysisPaths
from apantias.io.zarr import h5_to_zarr
from apantias.preflight_check import PreflightValidator
from apantias.settings import AppSettings, load_config
from apantias.transform.rechunk import rechunk_to_pixels

_logger = logging.getLogger(__name__)


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
        self._client, self._cluster = init(
            self.config.runtime.dask_temp, self.config.runtime.cpus, self.config.runtime.ram_mb
        )

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

        # ------------------------------------------------------------------
        # Stage 1: convert the raw HDF5 input into frame-chunked Zarr and
        # rechunk it into pixel-chunked Zarr for the per-pixel stages.
        # ------------------------------------------------------------------
        _logger.info("Start writing bin to zarr store.")
        h5_to_zarr(self.h5_path, self.paths.raw_framewise, self.config.frame, self.config.analysis)
        rechunk_to_pixels(self.paths.raw_framewise, self.paths.raw_pixelwise)
        _logger.info("Done.")

        # ------------------------------------------------------------------
        # Stage 2: declare the entire analysis pipeline.  No Dask work runs
        # here; only the task graph is built.
        # ------------------------------------------------------------------
        data_f, data_p = self._load_inputs()
        pipeline = self._build_pipeline(data_f, data_p)
        _logger.info("\n%s", pipeline.summary())

        # ------------------------------------------------------------------
        # Stage 3: materialise branch points.  This is the first place where
        # real computation is triggered.
        # ------------------------------------------------------------------
        _logger.info("Persisting branch points.")
        pipeline.materialize_branches(self._persist_or_store)

        # ------------------------------------------------------------------
        # Stage 4: write the remaining final outputs (archived arrays that
        # are not needed by later stages).
        # ------------------------------------------------------------------
        self._save_final_outputs(pipeline)

        # ------------------------------------------------------------------
        # Stage 5: copy every archived Zarr output into the final HDF5 archive.
        # ------------------------------------------------------------------
        _logger.info("Writing HDF5 archive.")
        write_analysis_outputs_to_h5(
            pipeline,
            self.config.analysis.h5_archive,
            chunk_threshold_bytes=self.config.frame.target_chunk_bytes,
        )

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

    def _build_pipeline(self, data_f: da.Array, data_p: da.Array) -> Pipeline:
        """Declare every output of the standard analysis pipeline.

        Each ``declare`` call registers the dask array together with metadata
        that controls where it is stored and whether it is a branch point.  No
        computation happens here; only the dask task graph is assembled.
        """
        p = Pipeline(self.config.analysis)

        _logger.info("Start calculating median.")
        p.declare(
            "median",
            stats.compute_median(data_p),
            "Pixel-wise median across frames and repetitions",
        )

        _logger.info("Start applying offset.")
        offset = self._load_offset()
        p.declare(
            "offset_corr",
            stats.compute_offset_corr(data_f, offset, p["median"]),
            "Raw frames corrected by external offset and pixel median",
            tags=[BRANCH],
        )

        _logger.info("Start common mode correction.")
        p.declare(
            "common_modes",
            stats.compute_common_modes(p["offset_corr"]),
            "Common-mode signal per frame, row and repetition",
            tags=[BRANCH],
        )
        p.declare(
            "signals",
            stats.subtract(p["offset_corr"], p["common_modes"]),
            "Offset- and common-mode-corrected detector signal",
            tags=[BRANCH],
        )

        _logger.info("Start calculating slopes, msd and mean signals.")
        p.declare(
            "slopes",
            stats.compute_slopes(p["signals"]),
            "Linear slope of signal versus repetition index",
        )
        p.declare(
            "msd",
            stats.compute_msd(p["signals"]),
            "Mean squared deviation of the corrected signal",
        )
        p.declare(
            "signals_mean",
            stats.compute_signals_mean(p["signals"]),
            "Mean corrected signal per frame and pixel",
        )

        _logger.info("Start pixelwise calculation.")
        p.declare(
            "pixelwise",
            apply_pixelwise(data_p, lambda t: t.max() - t.min()),
            "Per-pixel max-min range of raw signal",
        )

        return p

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

    def _save_final_outputs(self, pipeline: Pipeline) -> None:
        """Write all final (non-branch) archived outputs to their Zarr stores."""
        stages = pipeline.final_outputs()
        outputs = [s.array for s in stages]
        paths = [s.path for s in stages]

        _logger.info("Writing final outputs: %s", ", ".join(s.name for s in stages))
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
