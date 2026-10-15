"""End-to-end integration test for StandardAnalysis on tiny synthetic data."""

# Spawning a real Dask cluster and Zarr stores is integration territory; the
# library objects involved are only partially typed, so relax strict checks here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import pytest
import zarr

from apantias.settings import AnalysisSettings, AppSettings, RuntimeSettings
from apantias.standard import AnalysisPaths, StandardAnalysis

N_FRAMES, N_COLS, N_REPS, N_ROWS = 12, 4, 6, 4


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
        runtime=RuntimeSettings(cpus=2, ram_mb=1024, dask_temp=tmp_path / "dask_temp"),
        analysis=AnalysisSettings(
            h5_file=sample_h5_file,
            zarr_data=tmp_path / "data.zarr",
            zarr_temp=tmp_path / "temp.zarr",
            h5_archive=tmp_path / "archive.h5",
        ),
    )


def test_standard_analysis_end_to_end(config: AppSettings, tmp_path: Path) -> None:
    """Run StandardAnalysis end-to-end and verify every output store."""
    analysis = StandardAnalysis(config)
    analysis.run()

    paths = AnalysisPaths.from_settings(config.analysis)

    # Output shapes implied by the pipeline:
    #   per-pixel results are (n_cols, n_rows); frame-level results keep the
    #   frame axis; rep-axis reductions drop it.
    expected_shapes: dict[str, tuple[int, ...]] = {
        "median": (N_COLS, N_ROWS),
        "offset_corr": (N_FRAMES, N_COLS, N_REPS, N_ROWS),
        "common_modes": (N_FRAMES, N_COLS, N_REPS),
        "signals": (N_FRAMES, N_COLS, N_REPS, N_ROWS),
        "slopes": (N_FRAMES, N_COLS, N_ROWS),
        "msd": (N_COLS, N_ROWS),
        "signals_mean": (N_FRAMES, N_COLS, N_ROWS),
        "pixelwise": (N_COLS, N_ROWS),
    }

    for name, expected_shape in expected_shapes.items():
        store = getattr(paths, name)
        assert store.exists(), f"Zarr output {name} missing at {store}"
        arr = zarr.open_array(str(store), mode="r")
        assert tuple(arr.shape) == expected_shape, f"{name}: expected {expected_shape}, got {tuple(arr.shape)}"
        assert np.isfinite(np.asarray(arr)).all(), f"{name} contains non-finite values"

    # The final HDF5 archive must contain every output with matching shapes.
    archive = Path(config.analysis.h5_archive)
    assert archive.exists(), "HDF5 archive was not created"
    with h5py.File(archive, "r") as f:
        for name, expected_shape in expected_shapes.items():
            assert name in f, f"HDF5 dataset {name} missing from archive"
            ds = f[name]
            assert tuple(ds.shape) == expected_shape, f"archive {name}: expected {expected_shape}, got {ds.shape}"


def test_h5_archive_values_match_zarr(config: AppSettings) -> None:
    """The archive datasets must be identical to the Zarr outputs."""
    StandardAnalysis(config).run()

    paths = AnalysisPaths.from_settings(config.analysis)
    with h5py.File(config.analysis.h5_archive, "r") as f:
        for name in ("median", "offset_corr", "common_modes", "signals", "slopes", "msd", "signals_mean", "pixelwise"):
            zarr_data = np.asarray(zarr.open_array(str(getattr(paths, name)), mode="r"))
            # h5py Dataset slicing is partially unknown to the stubs.
            h5_data: npt.NDArray[np.float64] = np.asarray(f[name][:])  # type: ignore[index]
            np.testing.assert_allclose(h5_data, zarr_data)
