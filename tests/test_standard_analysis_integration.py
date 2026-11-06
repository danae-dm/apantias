"""End-to-end integration test for StandardAnalysis on tiny synthetic data."""

# Spawning a real Dask cluster and Zarr stores is integration territory; the
# library objects involved are only partially typed, so relax strict checks here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import Any

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
import yaml

from apantias.pipelines import StandardPipeline
from apantias.settings import AnalysisSettings, AppSettings, FrameSettings, RuntimeSettings

N_FRAMES, N_COLS, N_REPS, N_ROWS = 12, 4, 6, 4


@pytest.fixture(autouse=True)
def _mock_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make integration tests deterministic and independent of the host machine."""
    monkeypatch.setattr("apantias.runtime.resources.get_resources", lambda: (4, 32768))


# Names and shapes of outputs that are written to the HDF5 archive.
# Note: median has shape (n_rows, n_reps, n_cols) because it's computed across frames (axis 0)
# Only stages marked with is_archived=True are written to the HDF5 archive
EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "median": (N_ROWS, N_REPS, N_COLS),
    "signals_corr": (N_FRAMES, N_ROWS, N_REPS, N_COLS),
    "slopes": (N_FRAMES, N_ROWS, N_COLS),
    "signals_slope_corr": (N_FRAMES, N_ROWS, N_REPS, N_COLS),
    "signals": (N_FRAMES, N_ROWS, N_COLS),
}


@pytest.fixture
def sample_h5_file(tmp_path: Path) -> Path:
    """Create a tiny synthetic HDF5 input file.

    Layout follows the pipeline convention (n_frames, n_rows, n_reps, n_cols);
    n_cols == n_rows here so the row/col naming ambiguity in the historical
    layout comments cannot mask a real transpose bug.
    """
    h5_path = tmp_path / "input.h5"
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=(N_FRAMES, N_ROWS, N_REPS, N_COLS), dtype=np.uint16)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=data)
    return h5_path


@pytest.fixture
def config(tmp_path: Path, sample_h5_file: Path) -> AppSettings:
    """Build a fully tmp_path-scoped configuration.

    AppSettings is a frozen Pydantic model, so paths are provided at
    construction time rather than assigned afterwards.
    """
    return AppSettings(
        runtime=RuntimeSettings(dask_temp=tmp_path / "dask_temp"),
        frame=FrameSettings(
            n_frames=N_FRAMES,
            n_rows=N_ROWS,
            n_reps=N_REPS,
            n_cols=N_COLS,
        ),
        analysis=AnalysisSettings(
            h5_file=sample_h5_file,
            zarr_temp=tmp_path / "temp.zarr",
            h5_archive=tmp_path / "archive.h5",
            slope_threshold=0.0,
        ),
    )


# The Zarr store is removed by ``StandardPipeline._cleanup`` after archiving,
# so the integration tests validate the persistent HDF5 archive rather than
# the transient Zarr outputs.


def _with_keep_zarr_temp(config: AppSettings, keep: bool) -> AppSettings:
    """Return a copy of *config* with ``analysis.keep_zarr_temp`` set to *keep*."""
    return config.model_copy(update={"analysis": config.analysis.model_copy(update={"keep_zarr_temp": keep})})


def test_h5_archive_contents(config: AppSettings) -> None:
    """The archive must contain every output, the settings, and metadata."""
    StandardPipeline(config).run()

    archive_path = Path(config.analysis.h5_archive)
    assert archive_path.exists(), "HDF5 archive was not created"

    with h5py.File(archive_path, "r") as f:
        for name, expected_shape in EXPECTED_SHAPES.items():
            assert name in f, f"HDF5 dataset {name} missing from archive"
            ds: Any = f[name]
            assert tuple(ds.shape) == expected_shape, f"archive {name}: expected {expected_shape}, got {ds.shape}"
            # Every product documents itself via its stage description.
            assert "description" in ds.attrs, f"archive {name} lacks a description attribute"
            # All archived values must be finite.
            assert np.isfinite(np.asarray(ds[:])).all(), f"archive {name} contains non-finite values"

        # The full run configuration is embedded as YAML so the archive is
        # self-documenting without the original config file.
        assert "settings" in f, "settings dataset missing from archive"
        # h5py stores the YAML text as a variable-length string; depending on
        # the h5py version it is read back as str or bytes.
        settings: Any = f["settings"]
        raw = settings[()]
        settings_text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        embedded = yaml.safe_load(settings_text)
        assert embedded["analysis"]["h5_file"] == str(config.analysis.h5_file)
        assert embedded["runtime"]["dask_temp"] == str(config.runtime.dask_temp)


def test_keep_zarr_temp_preserves_store(config: AppSettings) -> None:
    """When keep_zarr_temp is True, the temporary Zarr store survives cleanup."""
    config = _with_keep_zarr_temp(config, True)
    StandardPipeline(config).run()

    temp_root = Path(config.analysis.zarr_temp)
    assert temp_root.is_dir(), "Temporary Zarr store was removed despite keep_zarr_temp=True"
    assert (temp_root / "median").exists(), "Expected Zarr output missing from preserved store"


def test_default_cleanup_removes_store(config: AppSettings) -> None:
    """By default the temporary Zarr store is removed after archiving."""
    StandardPipeline(config).run()
    assert not Path(config.analysis.zarr_temp).is_dir(), "Temporary Zarr store was not cleaned up"


def test_h5_archive_with_sign_flip_true(tmp_path: Path, sample_h5_file: Path) -> None:
    """Test StandardPipeline with raw_sign_flip=True (new default)."""
    config = AppSettings(
        runtime=RuntimeSettings(dask_temp=tmp_path / "dask_temp"),
        frame=FrameSettings(
            n_frames=N_FRAMES,
            n_rows=N_ROWS,
            n_reps=N_REPS,
            n_cols=N_COLS,
        ),
        analysis=AnalysisSettings(
            h5_file=sample_h5_file,
            zarr_temp=tmp_path / "temp.zarr",
            h5_archive=tmp_path / "archive.h5",
            raw_sign_flip=True,  # New default
            slope_threshold=0.0,
        ),
    )

    StandardPipeline(config).run()

    archive_path = Path(config.analysis.h5_archive)
    assert archive_path.exists(), "HDF5 archive was not created"

    with h5py.File(archive_path, "r") as f:
        # Verify archived outputs are finite and have expected shapes
        for name, expected_shape in EXPECTED_SHAPES.items():
            assert name in f, f"HDF5 dataset {name} missing from archive"
            ds: Any = f[name]
            assert tuple(ds.shape) == expected_shape, f"archive {name}: expected {expected_shape}, got {ds.shape}"
            # All archived values must be finite
            assert np.isfinite(np.asarray(ds[:])).all(), f"archive {name} contains non-finite values"


def test_h5_archive_with_sign_flip_false(tmp_path: Path, sample_h5_file: Path) -> None:
    """Test StandardPipeline with raw_sign_flip=False for backward compatibility."""
    config = AppSettings(
        runtime=RuntimeSettings(dask_temp=tmp_path / "dask_temp"),
        frame=FrameSettings(
            n_frames=N_FRAMES,
            n_rows=N_ROWS,
            n_reps=N_REPS,
            n_cols=N_COLS,
        ),
        analysis=AnalysisSettings(
            h5_file=sample_h5_file,
            zarr_temp=tmp_path / "temp.zarr",
            h5_archive=tmp_path / "archive.h5",
            raw_sign_flip=False,  # Old behavior: no sign flip
            slope_threshold=0.0,
        ),
    )

    StandardPipeline(config).run()

    archive_path = Path(config.analysis.h5_archive)
    assert archive_path.exists(), "HDF5 archive was not created"

    with h5py.File(archive_path, "r") as f:
        # Verify archived outputs are finite and have expected shapes
        for name, expected_shape in EXPECTED_SHAPES.items():
            assert name in f, f"HDF5 dataset {name} missing from archive"
            ds: Any = f[name]
            assert tuple(ds.shape) == expected_shape, f"archive {name}: expected {expected_shape}, got {ds.shape}"
            # All archived values must be finite
            assert np.isfinite(np.asarray(ds[:])).all(), f"archive {name} contains non-finite values"
