"""End-to-end integration test for StandardAnalysis on tiny synthetic data."""

# Spawning a real Dask cluster and Zarr stores is integration territory; the
# library objects involved are only partially typed, so relax strict checks here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import pytest
import yaml
import zarr

from apantias.settings import AnalysisSettings, AppSettings, RuntimeSettings
from apantias.standard import StandardAnalysis

N_FRAMES, N_COLS, N_REPS, N_ROWS = 12, 4, 6, 4

# Names and shapes of every product of the standard pipeline.  The on-disk
# layout is documented as <zarr_temp>/<stage name>; this test intentionally
# resolves the paths from the raw settings instead of from AnalysisPaths or
# Pipeline, so it independently verifies the documented contract.
EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "median": (N_COLS, N_ROWS),
    "offset_corr": (N_FRAMES, N_COLS, N_REPS, N_ROWS),
    "common_modes": (N_FRAMES, N_COLS, N_REPS),
    "signals": (N_FRAMES, N_COLS, N_REPS, N_ROWS),
    "slopes": (N_FRAMES, N_COLS, N_ROWS),
    "msd": (N_COLS, N_ROWS),
    "signals_mean": (N_FRAMES, N_COLS, N_ROWS),
    "pixelwise": (N_COLS, N_ROWS),
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
        runtime=RuntimeSettings(cpus=2, ram_mb=1024, dask_temp=tmp_path / "dask_temp"),
        analysis=AnalysisSettings(
            h5_file=sample_h5_file,
            zarr_data=tmp_path / "data.zarr",
            zarr_temp=tmp_path / "temp.zarr",
            h5_archive=tmp_path / "archive.h5",
        ),
    )


def _zarr_store(config: AppSettings, name: str) -> Path:
    """Documented on-disk layout of an analysis product."""
    return Path(config.analysis.zarr_temp) / name


def test_standard_analysis_end_to_end(config: AppSettings) -> None:
    """Run StandardAnalysis end-to-end and verify every output store."""
    analysis = StandardAnalysis(config)
    analysis.run()

    for name, expected_shape in EXPECTED_SHAPES.items():
        store = _zarr_store(config, name)
        assert store.exists(), f"Zarr output {name} missing at {store}"
        arr = zarr.open_array(str(store), mode="r")
        assert tuple(arr.shape) == expected_shape, f"{name}: expected {expected_shape}, got {tuple(arr.shape)}"
        assert np.isfinite(np.asarray(arr)).all(), f"{name} contains non-finite values"


def test_h5_archive_contents(config: AppSettings) -> None:
    """The archive must contain every output, the settings, and metadata."""
    StandardAnalysis(config).run()

    archive_path = Path(config.analysis.h5_archive)
    assert archive_path.exists(), "HDF5 archive was not created"

    with h5py.File(archive_path, "r") as f:
        for name, expected_shape in EXPECTED_SHAPES.items():
            assert name in f, f"HDF5 dataset {name} missing from archive"
            ds = f[name]
            assert tuple(ds.shape) == expected_shape, f"archive {name}: expected {expected_shape}, got {ds.shape}"
            # Every product documents itself via its stage description.
            assert "description" in ds.attrs, f"archive {name} lacks a description attribute"

        # The full run configuration is embedded as YAML so the archive is
        # self-documenting without the original config file.
        assert "settings" in f, "settings dataset missing from archive"
        # h5py stores the YAML text as a variable-length string; depending on
        # the h5py version it is read back as str or bytes.
        raw = f["settings"][()]  # type: ignore[index]
        settings_text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        embedded = yaml.safe_load(settings_text)
        assert embedded["analysis"]["h5_file"] == str(config.analysis.h5_file)
        assert embedded["runtime"]["dask_temp"] == str(config.runtime.dask_temp)


def test_h5_archive_values_match_zarr(config: AppSettings) -> None:
    """The archive datasets must be identical to the Zarr outputs."""
    StandardAnalysis(config).run()

    with h5py.File(config.analysis.h5_archive, "r") as f:
        for name in EXPECTED_SHAPES:
            zarr_data = np.asarray(zarr.open_array(str(_zarr_store(config, name)), mode="r"))
            # h5py Dataset slicing is partially unknown to the stubs.
            h5_data: npt.NDArray[np.float64] = np.asarray(f[name][:])  # type: ignore[index]
            np.testing.assert_allclose(h5_data, zarr_data)
