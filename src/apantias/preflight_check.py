"""Preflight validation and parameter inference for apantias analysis pipelines.

This module inspects the user configuration and input HDF5 dataset before any
heavy computational resources (such as Dask clusters or Zarr stores) are allocated.

Key responsibilities:
    1. Inspects the HDF5 input file and confirms it contains exactly one 4D dataset.
    2. Infers frame dimensions (n_frames, n_rows, n_cols, n_reps) from the shape.
    3. Validates analysis boundaries (frames_range, nreps_range, external offset).
    4. Prints and logs a clear summary of the discovered dataset and applied settings.
    5. Fails fast with an aggregated error report if any inconsistencies exist.

How to add more checks:
    1. Define a check method on `PreflightValidator`, e.g. `check_disk_space(self) -> None`.
    2. Inspect `self.ctx.config` or dataset metadata (`self.ctx.dataset_shape`, etc.).
    3. On failure, append a message via `self.ctx.add_error("Reason")`.
    4. Register your method in the `checks` list in `validate()`.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py

from apantias.settings import AppSettings

_logger = logging.getLogger(__name__)


class PreflightError(Exception):
    """Raised when preflight checks fail before running analysis."""

    pass


@dataclass
class PreflightContext:
    """Shared state across all preflight validation checks."""

    config: AppSettings
    errors: list[str] = field(default_factory=list)
    dataset_name: str | None = None
    dataset_shape: tuple[int, ...] | None = None
    dataset_dtype: str | None = None

    def add_error(self, message: str) -> None:
        """Record an error without raising immediately, allowing batch reporting."""
        self.errors.append(message)


class PreflightValidator:
    """Validates configuration and input files prior to starting analysis."""

    def __init__(self, config: AppSettings) -> None:
        self.ctx = PreflightContext(config=config)

    def validate(self, *, quiet: bool = False) -> AppSettings:
        """Run all registered checks, report findings, and return enriched AppSettings.

        Args:
            quiet: If True, suppresses console summary printing (still logs at INFO).

        Raises:
            PreflightError: If one or more validation checks fail.

        Returns:
            AppSettings: Copy of config with `frame` settings populated from the HDF5 shape.
        """
        checks = [
            self.check_h5_file_exists,
            self.check_single_dataset_and_shape,
            self.check_ranges_consistency,
            self.check_external_offset,
            # Add future checks here
        ]

        for check in checks:
            check()
            # Stop early only on fatal prerequisite failures (file missing or unreadable)
            # to avoid cascading attribute errors in subsequent range checks.
            if self.ctx.errors and check in (self.check_h5_file_exists, self.check_single_dataset_and_shape):
                break

        # Aggregate all errors and fail fast before any resources are initialized
        if self.ctx.errors:
            bullets = "\n  - ".join(self.ctx.errors)
            raise PreflightError(f"Preflight validation failed with {len(self.ctx.errors)} error(s):\n  - {bullets}")

        # Update settings with discovered dimensions (including n_frames)
        enriched_config = self._populate_frame_settings()

        # Display discovered shape and applied parameters to the user
        if not quiet:
            self._print_summary(enriched_config)

        return enriched_config

    def check_h5_file_exists(self) -> None:
        """Verify the specified raw HDF5 input file exists on disk."""
        path = Path(self.ctx.config.analysis.h5_file)
        if not path.is_file():
            self.ctx.add_error(f"HDF5 input file not found: {path}")

    def check_single_dataset_and_shape(self) -> None:
        """Verify the HDF5 file contains exactly one 4D dataset and read its metadata."""
        path = Path(self.ctx.config.analysis.h5_file)
        if not path.is_file():
            return

        try:
            # Inspect metadata only (read-only mode; data is not loaded into RAM)
            with h5py.File(path, "r") as f:
                datasets: list[str] = []

                def visitor(name: str, obj: object) -> None:
                    if isinstance(obj, h5py.Dataset):
                        datasets.append(name)

                f.visititems(visitor)

                if len(datasets) == 0:
                    self.ctx.add_error(f"HDF5 file '{path}' contains no datasets.")
                    return
                if len(datasets) > 1:
                    self.ctx.add_error(
                        f"HDF5 file '{path}' must contain exactly 1 dataset, but found {len(datasets)}: {datasets}"
                    )
                    return

                ds_name = datasets[0]
                ds = f[ds_name]
                if not isinstance(ds, h5py.Dataset):
                    self.ctx.add_error(f"Object '{ds_name}' is not an HDF5 Dataset.")
                    return

                self.ctx.dataset_name = ds_name
                self.ctx.dataset_shape = ds.shape
                self.ctx.dataset_dtype = str(ds.dtype)

                # Expect 4D: (n_frames, n_rows, n_reps, n_cols)
                if len(ds.shape) != 4:
                    self.ctx.add_error(
                        f"Dataset '{ds_name}' must be 4-dimensional (n_frames, n_rows, n_reps, n_cols), "
                        f"got shape {ds.shape} with {len(ds.shape)} dimensions."
                    )
        except Exception as exc:
            self.ctx.add_error(f"Failed to open or inspect HDF5 file '{path}': {exc}")

    def check_ranges_consistency(self) -> None:
        """Verify analysis ranges fall within the dataset dimensions and select > 0 items."""
        if self.ctx.dataset_shape is None or len(self.ctx.dataset_shape) != 4:
            return

        n_frames, _n_rows, raw_n_reps, _n_cols = self.ctx.dataset_shape
        analysis = self.ctx.config.analysis

        # Check frames_range boundaries
        eff_frames_stop = analysis.frames_range.resolve_stop(n_frames)
        if analysis.frames_range.start < 0 or analysis.frames_range.start >= n_frames:
            self.ctx.add_error(
                f"frames_range.start ({analysis.frames_range.start}) out of bounds for {n_frames} frames."
            )
        if analysis.frames_range.stop < -1:
            self.ctx.add_error(
                f"frames_range.stop ({analysis.frames_range.stop}) must be >= -1 (-1 selects end of dataset)."
            )
        elif eff_frames_stop > n_frames:
            self.ctx.add_error(
                f"frames_range.stop ({analysis.frames_range.stop}) exceeds total frames in dataset ({n_frames})."
            )
        elif analysis.frames_range.start >= eff_frames_stop:
            self.ctx.add_error(
                f"frames_range selects 0 frames "
                f"(start={analysis.frames_range.start}, stop={analysis.frames_range.stop})."
            )

        # Check nreps_range boundaries
        eff_reps_stop = analysis.nreps_range.resolve_stop(raw_n_reps)
        if analysis.nreps_range.start < 0 or analysis.nreps_range.start >= raw_n_reps:
            self.ctx.add_error(
                f"nreps_range.start ({analysis.nreps_range.start}) out of bounds for {raw_n_reps} repetitions."
            )
        if analysis.nreps_range.stop < -1:
            self.ctx.add_error(
                f"nreps_range.stop ({analysis.nreps_range.stop}) must be >= -1 (-1 selects end of dataset)."
            )
        elif eff_reps_stop > raw_n_reps:
            self.ctx.add_error(
                f"nreps_range.stop ({analysis.nreps_range.stop}) exceeds total repetitions in dataset ({raw_n_reps})."
            )
        elif analysis.nreps_range.start >= eff_reps_stop:
            self.ctx.add_error(
                f"nreps_range selects 0 repetitions "
                f"(start={analysis.nreps_range.start}, stop={analysis.nreps_range.stop})."
            )

    def check_external_offset(self) -> None:
        """Verify the external offset path exists if configured."""
        ext = self.ctx.config.analysis.ext_offset
        if ext is not None and str(ext) != "None":
            ext_path = Path(ext)
            if not ext_path.exists():
                self.ctx.add_error(f"External offset path does not exist: {ext_path}")

    def _populate_frame_settings(self) -> AppSettings:
        """Create an AppSettings copy populated with all 4 dimensions and resolved slice ranges."""
        assert self.ctx.dataset_shape is not None, "Dataset shape must be resolved before populating frame settings."
        n_frames, n_rows, n_reps, n_cols = self.ctx.dataset_shape

        new_frame = self.ctx.config.frame.model_copy(
            update={
                "n_frames": n_frames,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_reps": n_reps,
            }
        )

        # Concrete stop values for frames_range and nreps_range
        resolved_analysis = self.ctx.config.analysis.model_copy(
            update={
                "frames_range": self.ctx.config.analysis.frames_range.model_copy(
                    update={"stop": self.ctx.config.analysis.frames_range.resolve_stop(n_frames)}
                ),
                "nreps_range": self.ctx.config.analysis.nreps_range.model_copy(
                    update={"stop": self.ctx.config.analysis.nreps_range.resolve_stop(n_reps)}
                ),
            }
        )

        return self.ctx.config.model_copy(update={"frame": new_frame, "analysis": resolved_analysis})

    def _print_summary(self, config: AppSettings) -> None:
        """Print and log a summary of the discovered dataset and applied parameters."""
        assert self.ctx.dataset_shape is not None
        n_frames, n_rows, n_reps, n_cols = self.ctx.dataset_shape
        analysis = config.analysis

        n_selected_frames = analysis.frames_range.count(n_frames)
        n_selected_reps = analysis.nreps_range.count(n_reps)

        summary_lines = [
            "\n" + "=" * 50,
            f"Preflight Check: {analysis.h5_file}",
            "=" * 50,
            f"  Dataset:     '{self.ctx.dataset_name}' (dtype: {self.ctx.dataset_dtype})",
            f"  H5 Shape:    (n_frames={n_frames}, n_rows={n_rows}, n_reps={n_reps}, n_cols={n_cols})",
            "",
            "Frame Settings Applied:",
            f"  n_frames:    {config.frame.n_frames}",
            f"  n_rows:      {config.frame.n_rows}",
            f"  n_cols:      {config.frame.n_cols}",
            f"  n_reps:      {config.frame.n_reps}",
            "",
            "Analysis Slices:",
            f"  Frames:      {analysis.frames_range.start} to {analysis.frames_range.stop} "
            f"(of {n_frames} total, step={analysis.frames_range.step}) -> {n_selected_frames} selected",
            f"  Reps:        {analysis.nreps_range.start} to {analysis.nreps_range.stop} "
            f"(of {n_reps} total, step={analysis.nreps_range.step}) -> {n_selected_reps} selected",
            "=" * 50 + "\n",
        ]
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        _logger.info("Preflight check passed for %s: shape %s", analysis.h5_file, self.ctx.dataset_shape)
