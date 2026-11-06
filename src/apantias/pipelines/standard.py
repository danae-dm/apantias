"""Simplified pipeline declarations for standard analysis.

Each pipeline stage is fully self-contained:
- Inputs: which upstream stages it depends on
- Computation: how to transform inputs to outputs
- Output: where to write the result (auto-generated from name)
- Description: metadata written to archive
- Archive: whether to copy to HDF5 archive

The pipeline runs stages in dependency order. Each stage:
1. Loads inputs from their Zarr paths
2. Builds the computation graph
3. Writes output to its Zarr path (chunked according to current strategy)
4. Returns a lazy handle to the written data

Memory is bounded by the chunk size - workers process one chunk at a time.
No intermediate persistence in worker memory.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np

from apantias.processing import stats
from apantias.storage.outputs import ARCHIVE

if TYPE_CHECKING:
    from dask.distributed import Client, LocalCluster

    from apantias.settings import AppSettings
    from apantias.storage.outputs import Pipeline

_logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """A single stage in the analysis pipeline.

    Each stage is self-contained: it declares what inputs it needs,
    how to compute its output, and where to write the result.
    """

    name: str
    """Unique identifier for this stage. Also used to generate output_path."""

    inputs: list[str]
    """Names of upstream stages this depends on."""

    compute_fn: Callable[..., da.Array]
    """Function that computes the result. Can take any arguments."""

    description: str
    """Human-readable description written as metadata to the archive."""

    is_archived: bool = False
    """If True, this output will be copied to the HDF5 archive."""


class StandardPipeline:
    """Runs the standard analysis pipeline.

    Each pipeline stage is declared once, fully specified, and executed
    independently. No intermediate persistence - everything goes to Zarr.

    Args:
        config: Validated AppSettings with analysis parameters.
        client: Optional Dask client. If provided, the pipeline runs on this
                distributed cluster. If not provided, a local cluster is
                created and cleaned up internally.
    """

    def __init__(self, config: AppSettings, client: Client | None = None) -> None:
        from apantias.storage.paths import AnalysisPaths

        self.config = config
        self.paths = AnalysisPaths.from_settings(self.config.analysis)
        self._provided_client: Client | None = client
        self._internal_client: Client | None = None
        self._internal_cluster: LocalCluster | None = None

    def run(self) -> list[str]:
        """Run all pipeline stages and return list of archived stage names."""
        from apantias.processing.rechunk import rechunk_to_pixels as _rechunk
        from apantias.runtime.cluster import init as _init_cluster
        from apantias.storage.hdf5 import write_analysis_outputs_to_h5
        from apantias.storage.zarr import h5_to_zarr

        # Handle cluster setup/teardown
        if self._provided_client is None:
            self._internal_client, self._internal_cluster = _init_cluster(
                self.config.runtime.dask_temp,
                min_worker_memory_mb=self.config.runtime.min_worker_memory_mb,
                dashboard=self.config.runtime.dask_dashboard,
            )
            client: Client = self._internal_client  # type: ignore[assignment]
        else:
            client = self._provided_client

        try:
            t0 = time.perf_counter()
            # Stage 0: Convert raw HDF5 to Zarr (frame-chunked and pixel-chunked)
            _logger.info("Converting HDF5 to Zarr...")
            h5_to_zarr(
                Path(self.config.analysis.h5_file),
                self.paths.raw_framewise,
                self.config.frame,
                self.config.analysis,
                store_batch_size=1,  # Sequential for simplicity
            )
            _rechunk(self.paths.raw_framewise, self.paths.raw_pixelwise)
            _logger.info("Stage 0 (HDF5->Zarr + rechunk) finished in %.2fs", time.perf_counter() - t0)

            # Stage 1: Build and execute pipeline stages
            t0 = time.perf_counter()
            stages = self._declare_stages()
            self._execute_pipeline(stages, client)  # type: ignore[arg-type]
            _logger.info("Stage 1 (pipeline) finished in %.2fs", time.perf_counter() - t0)

            # Stage 2: Archive tagged outputs to HDF5
            t0 = time.perf_counter()
            archived = [s.name for s in stages if s.is_archived]
            _logger.info("Archiving stages: %s", archived)
            write_analysis_outputs_to_h5(
                self._to_pipeline_registry(stages),
                Path(self.config.analysis.h5_archive),
                config=self.config,
            )
            _logger.info("Stage 2 (archive) finished in %.2fs", time.perf_counter() - t0)

            return archived
        finally:
            self._cleanup()
            if self._internal_client is not None:
                self._internal_client.close()  # type: ignore[func-returns-value]
            if self._internal_cluster is not None:
                self._internal_cluster.close()  # type: ignore[func-returns-value]

    def _cleanup(self) -> None:
        """Remove the temporary Zarr store after archiving, unless configured to keep it."""
        if self.config.analysis.keep_zarr_temp:
            _logger.info("Keeping temp store: %s", self.paths.temp_root)
            return

        path = self.paths.temp_root
        if path.is_dir():
            import shutil

            try:
                shutil.rmtree(path)
                _logger.info("Removed temp store: %s", path)
            except OSError as exc:
                _logger.warning("Failed to remove temp store %s: %s", path, exc)

    def _declare_stages(self) -> list[PipelineStage]:
        """Declare all pipeline stages in dependency order."""
        # Order matters: dependencies must be declared before dependents
        return [
            # Pixel-wise median (from pixel-chunked data)
            PipelineStage(
                name="median",
                inputs=[],  # raw_pixelwise is written in Stage 0, not a pipeline stage
                compute_fn=lambda: stats.compute_median(da.from_zarr(self.paths.raw_pixelwise)),  # type: ignore[reportUnknownVariableType]
                description="Pixel-wise median across frames",
                is_archived=True,
            ),
            # Offset correction (frame-chunked)
            PipelineStage(
                name="offset_corr",
                inputs=["median"],
                compute_fn=lambda inputs: stats.compute_offset_corr(  # type: ignore[reportUnknownArgumentType, reportUnknownParameterType]
                    da.from_zarr(self.paths.raw_framewise),  # type: ignore[reportUnknownVariableType]
                    self._resolve_offset(inputs),  # type: ignore[reportUnknownArgumentType]
                ),
                description="Raw frames corrected by external offset or pixel median",
                is_archived=False,  # intermediate
            ),
            # Common mode correction (frame-chunked)
            PipelineStage(
                name="common_modes",
                inputs=["offset_corr"],
                compute_fn=lambda inputs: stats.compute_common_modes(  # type: ignore[reportUnknownArgumentType, reportUnknownParameterType]
                    inputs["offset_corr"],  # type: ignore[reportUnknownArgumentType]
                ),
                description="Common-mode signal per frame, row and repetition",
                is_archived=False,  # intermediate
            ),
            # Final signals (frame-chunked)
            PipelineStage(
                name="signals",
                inputs=["offset_corr", "common_modes"],
                compute_fn=lambda inputs: stats.subtract(  # type: ignore[reportUnknownArgumentType, reportUnknownParameterType]
                    inputs["offset_corr"],  # type: ignore[reportUnknownArgumentType]
                    inputs["common_modes"],  # type: ignore[reportUnknownArgumentType]
                ),
                description="Offset- and common-mode-corrected detector signal",
                is_archived=False,
            ),
            PipelineStage(
                name="slopes",
                inputs=["signals"],
                compute_fn=lambda inputs: stats.compute_slopes(  # type: ignore[reportUnknownArgumentType, reportUnknownParameterType]
                    inputs["signals"],  # type: ignore[reportUnknownArgumentType]
                ),
                description="Linear slope of signal versus repetition index",
                is_archived=True,
            ),
            PipelineStage(
                name="signals_mean",
                inputs=["signals"],
                compute_fn=lambda inputs: stats.compute_signals_mean(  # type: ignore[reportUnknownArgumentType, reportUnknownParameterType]
                    inputs["signals"],  # type: ignore[reportUnknownArgumentType]
                ),
                description="Mean across n_reps",
                is_archived=True,
            ),
        ]

    def _execute_pipeline(self, stages: list[PipelineStage], client: Client) -> None:
        """Execute all pipeline stages in dependency order."""
        results: dict[str, da.Array] = {}  # Cache already-executed stages

        for stage in stages:
            _logger.info("Executing stage: %s", stage.name)

            # Load inputs from Zarr (for stages that depend on other pipeline stages)
            for inp in stage.inputs:
                if inp not in results:
                    input_path = self.paths.temp_root / inp
                    results[inp] = da.from_zarr(input_path)  # type: ignore[reportUnknownVariableType]

            # Get input arrays from cache
            inputs = {inp: results[inp] for inp in stage.inputs}

            # Build and execute computation (call the lambda with inputs if needed)
            if stage.inputs:
                result = stage.compute_fn(inputs)
            else:
                result = stage.compute_fn()

            # Auto-generate output path from name
            output_path = self.paths.temp_root / stage.name

            # Write to Zarr (uses the provided Dask client)
            _logger.info("Writing %s to %s", stage.name, output_path)

            t0 = time.perf_counter()
            self._write_to_zarr(result, output_path, client)
            elapsed = time.perf_counter() - t0
            _logger.info('Stage "%s" finished in %.2fs', stage.name, elapsed)

            # Cache for downstream stages
            results[stage.name] = da.from_zarr(output_path)  # type: ignore[reportUnknownVariableType]

    def _write_to_zarr(self, arr: da.Array, path: Path, client: Client) -> None:
        """Write a dask array to Zarr, preserving current chunking strategy.

        Uses the provided Dask client for distributed computation.

        The chunk shape is preserved from the dask array. For 4D data with shape
        (n_frames, n_rows, n_reps, n_cols), the chunks will be a tuple of 4
        integers representing the chunk size along each axis.
        """
        import zarr

        # Open or create Zarr array with matching chunking
        # arr.chunks is a tuple of tuples, one per axis. We need the full shape.
        chunks = tuple(c[0] for c in arr.chunks)  # type: ignore[reportUnknownVariableType]
        target = zarr.open_array(  # type: ignore[reportUnknownVariableType]
            str(path),
            mode="w",
            shape=arr.shape,
            chunks=chunks,
            dtype=arr.dtype,  # type: ignore[reportUnknownMemberType]
            zarr_format=3,
        )

        # Write using da.store with the provided Dask client
        # da.store accepts a client argument for distributed execution
        da.store(arr, target, lock=False, compute_kwargs={"client": client})  # type: ignore[reportUnknownVariableType]

    def _load_offset(self) -> da.Array:
        """Load external offset if configured, otherwise return zeros.

        Offset has shape (n_rows, n_cols) to broadcast with the 4D data array
        of shape (n_frames, n_rows, n_reps, n_cols).
        """
        from apantias.storage.hdf5 import split_h5_dataset_path

        ext = self.config.analysis.ext_offset
        if ext is None or str(ext) == "None":
            return da.zeros(  # type: ignore[reportUnknownVariableType]
                (self.config.frame.n_rows, self.config.frame.n_cols),
                chunks=-1,
                dtype=np.float64,
            )

        # Check if it's an HDF5 reference or Zarr path
        h5_ref = split_h5_dataset_path(ext)
        if h5_ref is not None:
            # HDF5 file: load as dask array (small, so eager reading is fine)
            from apantias.storage.hdf5 import load_h5_dataset

            file_path, dataset_name = h5_ref
            return load_h5_dataset(file_path, dataset_name)
        else:
            # Zarr path
            return da.from_zarr(ext)  # type: ignore[reportUnknownVariableType]

    def _resolve_offset(self, inputs: dict[str, da.Array]) -> da.Array:
        """Return external offset if configured, otherwise return the median.

        Args:
            inputs: Dictionary of upstream stage outputs, must contain 'median'.

        Returns:
            Offset array with shape (n_rows, n_cols) or (n_rows, n_reps, n_cols).
            If ext_offset is configured, load it and return. Otherwise return the
            computed median from inputs.
        """
        ext = self.config.analysis.ext_offset
        if ext is not None and str(ext) != "None":
            # Load external offset
            return self._load_offset()
        # No external offset: use the median
        return inputs["median"]

    def _to_pipeline_registry(self, stages: list[PipelineStage]) -> Pipeline:
        """Convert stages to Pipeline registry for archiving."""
        from apantias.storage.outputs import Pipeline

        p = Pipeline(self.config.analysis)
        for stage in stages:
            # Re-create stage with disk-backed array
            arr = da.from_zarr(self.paths.temp_root / stage.name)  # type: ignore[reportUnknownVariableType]
            p.declare(
                name=stage.name,
                array=arr,
                description=stage.description,
                tags=[ARCHIVE] if stage.is_archived else [],
            )
        return p
