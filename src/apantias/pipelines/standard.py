# Standard analysis orchestrates Dask/Zarr calls; both libraries ship incomplete
# stubs, so strict-mode unknown-type checks are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

"""Definition of the standard data analysis pipeline.

This module declares the analysis stages in a ``Pipeline``.  Each stage is
registered together with metadata that controls where it is stored, whether
downstream stages consume it (branch point), and whether it ends up in the
final HDF5 archive.

Declaration and materialization are interleaved: every stage that later stages
consume is materialized (in-memory persist or Zarr) immediately after its
declaration, *before* the consuming stage's dask graph is built.  This is
required because dask graphs capture arrays by value at declaration time —
materializing after the fact cannot rewire an already-built graph.  Leaf
outputs stay lazy and are written together in one batched ``da.store``.

How a stage is archived is decided centrally by the HDF5 writer based on the
configured chunk threshold.  Infrastructure lives in dedicated modules and is
only wired together here: path layout in ``storage.paths``, the pipeline registry in
``storage.outputs``, stage materialization in ``_persist_or_store``, and archiving
in ``storage.hdf5``.
"""

import logging
import shutil
from pathlib import Path
from typing import cast

import dask.array as da
import zarr

from apantias.preflight import PreflightValidator
from apantias.processing import stats
from apantias.processing.rechunk import rechunk_to_pixels
from apantias.processing.reductions import apply_pixelwise
from apantias.runtime import init
from apantias.settings import AppSettings, load_config
from apantias.storage.hdf5 import write_analysis_outputs_to_h5
from apantias.storage.outputs import ARCHIVE, BRANCH, Pipeline
from apantias.storage.paths import AnalysisPaths
from apantias.storage.zarr import h5_to_zarr

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
            self.config.runtime.dask_temp,
            self.config.runtime.cpus,
            self.config.runtime.ram_mb,
            min_worker_memory_mb=self.config.runtime.min_worker_memory_mb,
            dashboard=self.config.runtime.dask_dashboard,
            use_nanny=self.config.runtime.dask_nanny,
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
            self._cleanup()
            client.close()
            cluster.close()

    def _cleanup(self) -> None:
        """Remove the temporary Zarr store after archiving, unless configured to keep it.

        Called in the finally block of ``run()`` so it always executes,
        even when the analysis fails after the archive step has completed.
        Errors during cleanup are logged but never raised — they cannot
        undo the analysis results already written to the HDF5 archive.
        """
        if self.config.analysis.keep_zarr_temp:
            _logger.info("Keeping temp store: %s", self.paths.temp_root)
            return

        path = self.paths.temp_root
        if path.is_dir():
            try:
                shutil.rmtree(path)
                _logger.info("Removed temp store: %s", path)
            except OSError as exc:
                _logger.warning("Failed to remove temp store %s: %s", path, exc)

    def _run_analysis(self) -> None:
        zarr.open_group(self.paths.temp_root, mode="a")

        # ------------------------------------------------------------------
        # Stage 1: convert the raw HDF5 input into frame-chunked Zarr and
        # rechunk it into pixel-chunked Zarr for the per-pixel stages.
        # ------------------------------------------------------------------
        _logger.info("Start writing bin to zarr store.")
        h5_to_zarr(self.h5_path, self.paths.raw_framewise, self.config.frame, self.config.analysis)
        rechunk_to_pixels(self.paths.raw_framewise, self.paths.raw_pixelwise)
        _logger.info("Done.")

        # ------------------------------------------------------------------
        # Stage 2: declare the pipeline stage by stage.  Every stage that is
        # consumed by a later stage is materialized right after its
        # declaration (see _build_pipeline), so this is also the first place
        # where real computation is triggered.
        # ------------------------------------------------------------------
        data_f, data_p = self._load_inputs()
        pipeline = self._build_pipeline(data_f, data_p)
        _logger.info("\n%s", pipeline.summary())

        # ------------------------------------------------------------------
        # Stage 3: write the remaining leaf outputs (archived arrays that no
        # later stage consumes) in one batched da.store.
        # ------------------------------------------------------------------
        self._save_final_outputs(pipeline)

        # ------------------------------------------------------------------
        # Stage 4: copy every archived Zarr output into the final HDF5 archive.
        # ------------------------------------------------------------------
        _logger.info("Writing HDF5 archive.")
        write_analysis_outputs_to_h5(
            pipeline,
            self.config.analysis.h5_archive,
            config=self.config,
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

        Stages are declared in dependency order.  A stage that later stages
        consume (a branch point) is materialized immediately after its
        declaration, *before* the consuming graph is built.  This ordering is
        essential: dask graphs capture arrays by value at ``declare`` time, so
        ``p["<name>"]`` hands the consuming stage whatever object the registry
        holds at that moment.  Materializing between declarations guarantees
        downstream graphs read the persisted array (cluster memory or Zarr)
        instead of embedding the full lazy upstream — without this, shared
        subgraphs are resubmitted to the scheduler several times (dask issue
        #9888 warnings) and correctness would rely on the cluster's key cache.

        Leaf outputs (nothing downstream consumes them) stay lazy and are
        written together in one batched ``da.store`` by ``_save_final_outputs``.
        """
        p = Pipeline(
            self.config.analysis,
            memory_budget_bytes=self._memory_budget_bytes(),
        )

        def materialize(name: str) -> None:
            """Persist stage *name* in place (memory if it fits, Zarr otherwise).

            Replaces the stage's lazy array with the materialized handle so
            subsequent ``p[name]`` lookups — and therefore all graphs built
            afterwards — reference the persisted version.
            """
            stage = p.stage(name)
            stage.array = self._persist_or_store(stage.array, stage.path, p)

        _logger.info("Start calculating median.")
        p.declare(
            "median",
            stats.compute_median(data_p),
            "Pixel-wise median across frames and repetitions",
            tags=[BRANCH, ARCHIVE],  # consumed by offset_corr
        )
        materialize("median")

        _logger.info("Start applying offset.")
        offset = self._load_offset()
        p.declare(
            "offset_corr",
            # p["median"] is the materialized median here, so this graph reads
            # the persisted store instead of recomputing the median from raw data.
            stats.compute_offset_corr(data_f, offset, p["median"]),
            "Raw frames corrected by external offset and pixel median",
            tags=[BRANCH, ARCHIVE],  # consumed by common_modes and signals
        )
        materialize("offset_corr")

        _logger.info("Start common mode correction.")
        p.declare(
            "common_modes",
            stats.compute_common_modes(p["offset_corr"]),
            "Common-mode signal per frame, row and repetition",
            tags=[BRANCH, ARCHIVE],  # consumed by signals
        )
        materialize("common_modes")
        p.declare(
            "signals",
            stats.subtract(p["offset_corr"], p["common_modes"]),
            "Offset- and common-mode-corrected detector signal",
            tags=[BRANCH, ARCHIVE],  # consumed by slopes, msd and signals_mean
        )
        materialize("signals")

        _logger.info("Start calculating slopes, msd and mean signals.")
        # Leaf outputs: declared on top of the materialized signals, but left
        # lazy — they are stored in one batched da.store later.
        p.declare(
            "slopes",
            stats.compute_slopes(p["signals"]),
            "Linear slope of signal versus repetition index",
            tags=[ARCHIVE],
        )
        p.declare(
            "msd",
            stats.compute_msd(p["signals"]),
            "Mean squared deviation of the corrected signal",
            tags=[ARCHIVE],
        )
        p.declare(
            "signals_mean",
            stats.compute_signals_mean(p["signals"]),
            "Mean corrected signal per frame and pixel",
            tags=[ARCHIVE],
        )

        _logger.info("Start pixelwise calculation.")
        p.declare(
            "pixelwise",
            apply_pixelwise(data_p, lambda t: t.max() - t.min()),
            "Per-pixel max-min range of raw signal",
            tags=[ARCHIVE],
        )

        return p

    def _persist_or_store(self, arr: da.Array, path: Path, pipeline: Pipeline) -> da.Array:
        """Materialize a branch-point array so it is not recomputed.

        The array is persisted in worker memory only while the configured
        in-memory budget (``analysis.persist_memory_percent`` of total RAM) is
        not exceeded.  Otherwise it is written to Zarr and read back from disk.
        """
        if pipeline.can_persist_in_memory(arr.nbytes):
            _logger.info("Persist %s in memory (%.2f MB).", path.name, arr.nbytes / 1e6)
            persisted = cast(da.Array, self._client.persist(arr))
            # Persist in memory for downstream use and also write a Zarr copy:
            # the HDF5 archive pass reads archived stages from Zarr, and Zarr
            # serves as the fallback when the memory budget is exhausted.
            persisted.to_zarr(path, overwrite=True)
            pipeline.record_memory_use(arr.nbytes)
            return persisted

        _logger.info(
            "Store %s to zarr (%.2f MB) — memory budget exhausted.",
            path.name,
            arr.nbytes / 1e6,
        )
        arr.to_zarr(path, overwrite=True)
        return cast(da.Array, da.from_zarr(path))

    def _memory_budget_bytes(self) -> int:
        """Return the total in-memory budget for persisted branch points.

        The budget is ``analysis.persist_memory_percent`` percent of the
        configured total RAM.
        """
        ram_mb = self.config.runtime.ram_mb
        if ram_mb <= 0:
            return 0
        percent = self.config.analysis.persist_memory_percent
        return int(ram_mb * 1024 * 1024 * (percent / 100.0))

    def _save_final_outputs(self, pipeline: Pipeline) -> None:
        """Write all leaf outputs to their Zarr stores."""
        stages = pipeline.leaf_outputs()
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
        # Back-fill the registry with disk-backed handles: after this point
        # every stage of the pipeline refers to materialized data, so touching
        # a stage later cannot silently recompute the whole upstream graph.
        for stage in stages:
            stage.array = cast(da.Array, da.from_zarr(stage.path))
        _logger.info("Done.")
