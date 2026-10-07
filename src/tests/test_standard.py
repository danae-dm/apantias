"""Regression tests for ``StandardAnalysis.run()`` (src/apantias/standard.py).

Every ``fixtures/case_<name>/`` directory is one case: ``<name>.h5`` is the raw input (dataset ``raw_data``,
shape ``(frames, cols, nreps, rows)``) and ``expected_<name>.h5`` holds the golden outputs. See the "Testing"
section in README.md for details and for how to (re)generate golden files with ``--update-golden``.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest
import yaml
import zarr

import apantias
from apantias.settings import AppSettings
from apantias.standard import StandardAnalysis

FIXTURES = Path(__file__).parent / "fixtures"
INPUT_DATASET = "raw_data"
# Arrays written to analysis.zarr_data. With full nreps/frames ranges they are the input data unchanged, so they
# are compared exactly against the input file instead of being duplicated into the golden file.
DATA_OUTPUTS = ("frame_chunked", "pixel_chunked")
# Everything in the temp working directory that run() is allowed to create. Anything else is a stray write.
WORKDIR_ENTRIES = ("config.yaml", "dask_temp", "zarr_data.zarr", "zarr_temp.zarr")

# Dask chunking does not depend on the worker count, so results are bit-identical between runs on one machine.
# The tolerance only absorbs float64 summation-order differences (SIMD/numpy builds) in the means/sums over at most
# frames*nreps = 5e4 values. Inputs are uint16 integers, so medians/differences are multiples of 0.5 and any real
# change is many orders of magnitude larger.
RTOL = 1e-10
ATOL = 1e-10
# Arrays are compared in slabs along axis 0 so the ~750 MB float64 outputs never have to be fully in memory.
SLAB_FRAMES = 50


@dataclass(frozen=True)
class Case:
    name: str
    directory: Path

    @property
    def input_h5(self) -> Path:
        return self.directory / f"{self.name}.h5"

    @property
    def expected_h5(self) -> Path:
        return self.directory / f"expected_{self.name}.h5"


def discover_cases() -> list[Case]:
    if not FIXTURES.is_dir():
        return []
    cases = []
    for directory in sorted(FIXTURES.iterdir()):
        if not directory.is_dir() or not directory.name.startswith("case_"):
            continue
        case = Case(directory.name.removeprefix("case_"), directory)
        h5_files = sorted(p.name for p in directory.glob("*.h5"))
        if not case.input_h5.is_file() or set(h5_files) - {case.input_h5.name, case.expected_h5.name}:
            raise ValueError(
                f"{directory}: expected '{case.input_h5.name}' and '{case.expected_h5.name}', found {h5_files}"
            )
        cases.append(case)
    return cases


CASES = discover_cases() or [
    pytest.param(None, marks=pytest.mark.skip(reason=f"no fixture cases in {FIXTURES} (directory is gitignored)"))
]


def write_case_config(case: Case, workdir: Path) -> Path:
    """Write the YAML config for a case.

    The frame layout and nreps/frames ranges are taken from the input file, so the whole input is analysed.
    All paths are absolute and point into ``workdir``, except the read-only input file.
    """
    with h5py.File(case.input_h5, "r") as f:
        n_frames, cols, nreps, rows = f[INPUT_DATASET].shape
    settings = AppSettings.model_validate({
        # cpus/ram_mb are fixed (0 would auto-detect); they only size the Dask cluster, not the results.
        "runtime": {"cpus": 4, "ram_mb": 4096, "dask_temp": workdir / "dask_temp"},
        "frame": {"rows": rows, "cols": cols, "nreps": nreps},
        "analysis": {
            "h5_file": case.input_h5,
            "zarr_data": workdir / "zarr_data.zarr",
            "zarr_temp": workdir / "zarr_temp.zarr",
            "h5_archive": workdir / "h5_archive",
            "ext_offset": None,
            "nreps_range": {"start": 0, "stop": nreps},
            "frames_range": {"start": 0, "stop": n_frames},
        },
    })
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(settings.model_dump(mode="json"), sort_keys=False))
    return path


def result_settings(config: AppSettings) -> str:
    """The settings that determine the analysis results (no paths/resources), stored in the golden file."""
    analysis = config.analysis
    return yaml.safe_dump({
        "frame": config.frame.model_dump(mode="json"),
        "nreps_range": analysis.nreps_range.model_dump(mode="json"),
        "frames_range": analysis.frames_range.model_dump(mode="json"),
        "ext_offset": None if analysis.ext_offset is None else str(analysis.ext_offset),
    })


def file_state(directory: Path) -> dict[str, tuple[int, int]]:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


def zarr_arrays(store: Path) -> dict[str, zarr.Array]:
    return {name: arr for name, arr in zarr.open_group(store, mode="r").arrays()}


def write_golden(path: Path, arrays: dict[str, zarr.Array], settings: str) -> None:
    partial = path.with_name(path.name + ".partial")
    with h5py.File(partial, "w") as f:
        f.attrs["settings"] = settings
        f.attrs["apantias_version"] = apantias.__version__
        for name, arr in sorted(arrays.items()):
            row_bytes = math.prod(arr.shape[1:]) * arr.dtype.itemsize
            chunks = (min(arr.shape[0], max(1, 2**20 // row_bytes)), *arr.shape[1:])
            ds = f.create_dataset(
                name,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=chunks,
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
            )
            for start in range(0, arr.shape[0], SLAB_FRAMES):
                ds[start : start + SLAB_FRAMES] = arr[start : start + SLAB_FRAMES]
    os.replace(partial, path)


def assert_array_matches(label: str, actual, expected, exact: bool) -> None:
    """Compare two array-likes slab by slab; on failure report the first differing index (in full-array coords)."""
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != expected {expected.shape}"
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype} != expected {expected.dtype}"
    for start in range(0, actual.shape[0], SLAB_FRAMES):
        stop = min(start + SLAB_FRAMES, actual.shape[0])
        a, e = np.asarray(actual[start:stop]), np.asarray(expected[start:stop])
        try:
            if exact:
                np.testing.assert_array_equal(a, e)
            else:
                np.testing.assert_allclose(a, e, rtol=RTOL, atol=ATOL)
        except AssertionError as exc:
            mismatch = (a != e) if exact else ~np.isclose(a, e, rtol=RTOL, atol=ATOL, equal_nan=True)
            idx = tuple(int(i) for i in np.argwhere(mismatch)[0])
            raise AssertionError(
                f"{label}: first difference at index {(idx[0] + start, *idx[1:])}: "
                f"actual={a[idx]!r} expected={e[idx]!r} (axis-0 slab {start}:{stop})\n{exc}"
            ) from exc


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name if c else "no-fixtures")
def test_run_regression(case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    update = request.config.getoption("--update-golden")
    if not update and not case.expected_h5.is_file():
        pytest.fail(f"[{case.name}] missing {case.expected_h5}; generate it with --update-golden after review")
    fixture_state = file_state(case.directory)

    config_path = write_case_config(case, tmp_path)
    ctx = f"[{case.name}] (config {config_path})"
    # Dask workers inherit the cwd; running inside tmp_path makes any stray relative-path write land there
    # (and fail the workdir check below) instead of polluting the repository.
    monkeypatch.chdir(tmp_path)

    analysis = StandardAnalysis(config_path)
    result = analysis.run()

    assert result is None, f"{ctx}: run() returned {result!r}; results are expected only in the zarr stores"
    assert file_state(case.directory) == fixture_state, f"{ctx}: run() modified files in {case.directory}"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(WORKDIR_ENTRIES), f"{ctx}: unexpected workdir"

    # zarr_data: raw data re-chunked framewise and pixelwise, must equal the input exactly.
    data_arrays = zarr_arrays(analysis.zarr_data)
    assert sorted(data_arrays) == sorted(DATA_OUTPUTS), f"{ctx}: arrays in {analysis.zarr_data}"
    with h5py.File(case.input_h5, "r") as f:
        for name in DATA_OUTPUTS:
            assert_array_matches(f"{ctx} zarr_data/{name}", data_arrays[name], f[INPUT_DATASET], exact=True)

    # zarr_temp: all analysis results, compared against the golden file.
    temp_arrays = zarr_arrays(analysis.temp_zarr)
    settings = result_settings(analysis.config)
    if update:
        write_golden(case.expected_h5, temp_arrays, settings)

    with h5py.File(case.expected_h5, "r") as golden:
        assert golden.attrs.get("settings") == settings, (
            f"{ctx}: {case.expected_h5.name} was generated with different settings; regenerate with --update-golden\n"
            f"golden:\n{golden.attrs.get('settings')}\ncurrent:\n{settings}"
        )
        assert sorted(temp_arrays) == sorted(golden), f"{ctx}: zarr_temp arrays differ from {case.expected_h5.name}"
        for name in sorted(golden):
            arr = temp_arrays[name]
            assert_array_matches(
                f"{ctx} zarr_temp/{name}", arr, golden[name], exact=not np.issubdtype(arr.dtype, np.floating)
            )
