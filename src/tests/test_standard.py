"""Regression check for ``StandardAnalysis.run()`` (src/apantias/standard.py), run manually before committing.

A case folder contains the raw input ``<name>.h5`` (dataset ``raw_data``, shape ``(frames, cols, nreps, rows)``)
and the expected output ``expected_<name>.h5``:

    uv run python src/tests/test_standard.py                     # all case_* folders in src/tests/fixtures
    uv run python src/tests/test_standard.py path/to/case_x ...  # selected folders
    uv run python src/tests/test_standard.py --update ...        # (over)write expected_<name>.h5 from current output
"""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import yaml
import zarr

from apantias.settings import AppSettings
from apantias.standard import StandardAnalysis

FIXTURES = Path(__file__).parent / "fixtures"
# Results are normally bit-identical; the tolerance only absorbs float64 summation-order differences between
# machines. Inputs are uint16, so any real change is many orders of magnitude larger.
RTOL = ATOL = 1e-10


def _write_config(input_h5: Path, workdir: Path) -> Path:
    """Config analysing the whole input; everything run() writes goes into workdir."""
    with h5py.File(input_h5, "r") as f:
        n_frames, cols, nreps, rows = f["raw_data"].shape
    settings = AppSettings.model_validate({
        "runtime": {"cpus": 0, "ram_mb": 0, "dask_temp": workdir / "dask_temp"},
        "frame": {"rows": rows, "cols": cols, "nreps": nreps},
        "analysis": {
            "h5_file": input_h5,
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


def _compare(name: str, actual: zarr.Array, expected, exact: bool) -> None:
    """Raise an AssertionError naming the first differing index if the arrays differ."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: {actual.shape} {actual.dtype} != expected {expected.shape} {expected.dtype}")
    # Compare per zarr chunk along axis 0: bounded memory and every chunk is decompressed only once.
    step = actual.chunks[0]
    for start in range(0, actual.shape[0], step):
        a, e = actual[start : start + step], expected[start : start + step]
        if np.array_equal(a, e, equal_nan=not exact):
            continue
        ok = (a == e) if exact else np.isclose(a, e, rtol=RTOL, atol=ATOL, equal_nan=True)
        if not ok.all():
            idx = tuple(int(i) for i in np.argwhere(~ok)[0])
            raise AssertionError(
                f"{name}: {np.count_nonzero(~ok)} values differ in axis-0 range {start}:{start + len(a)}, "
                f"first at {(idx[0] + start, *idx[1:])}: actual={a[idx]!r} expected={e[idx]!r}"
            )


def _write_expected(path: Path, arrays: dict[str, zarr.Array]) -> None:
    partial = path.with_name(path.name + ".partial")
    with h5py.File(partial, "w") as f:
        for name, arr in sorted(arrays.items()):
            ds = f.create_dataset(
                name, shape=arr.shape, dtype=arr.dtype, chunks=True, **hdf5plugin.Blosc(cname="zstd", clevel=5)
            )
            step = arr.chunks[0]
            for start in range(0, arr.shape[0], step):
                ds[start : start + step] = arr[start : start + step]
    os.replace(partial, path)


def _run_case(folder: Path, update: bool) -> None:
    inputs = [p for p in folder.glob("*.h5") if not p.name.startswith("expected_")]
    if len(inputs) != 1:
        raise ValueError(f"expected exactly one input .h5 in {folder}, found {[p.name for p in inputs]}")
    input_h5 = inputs[0]
    expected_h5 = folder / f"expected_{input_h5.name}"
    if not update and not expected_h5.is_file():
        raise FileNotFoundError(f"{expected_h5} is missing, create it with --update")

    with tempfile.TemporaryDirectory() as tmp:
        analysis = StandardAnalysis(_write_config(input_h5, Path(tmp)))
        analysis.run()

        # zarr_data holds the raw data re-chunked framewise and pixelwise, so it must equal the input exactly.
        with h5py.File(input_h5, "r") as f:
            for name in ("frame_chunked", "pixel_chunked"):
                actual = zarr.open_array(analysis.zarr_data / name, mode="r")
                _compare(f"zarr_data/{name}", actual, f["raw_data"], exact=True)

        results = dict(zarr.open_group(analysis.temp_zarr, mode="r").arrays())
        if update:
            _write_expected(expected_h5, results)
        with h5py.File(expected_h5, "r") as f:
            if sorted(results) != sorted(f):
                raise AssertionError(f"outputs {sorted(results)} != expected {sorted(f)}")
            for name in sorted(f):
                exact = not np.issubdtype(f[name].dtype, np.floating)
                _compare(f"zarr_temp/{name}", results[name], f[name], exact)


def run_cases(folders: list[Path], update: bool = False) -> dict[str, str | None]:
    """Run StandardAnalysis on every case folder and compare its outputs with ``expected_<name>.h5``.

    With ``update=True`` the expected files are (over)written from the current output first.
    Returns ``{folder name: None if passed, else the failure reason}`` and prints a summary.
    """
    results: dict[str, str | None] = {}
    for folder in map(Path, folders):
        try:
            _run_case(folder, update)
            results[folder.name] = None
        except AssertionError as exc:
            results[folder.name] = str(exc)
        except Exception as exc:
            traceback.print_exc()
            results[folder.name] = f"{type(exc).__name__}: {exc}"

    print("\n" + "=" * 60)
    for name, failure in results.items():
        print(f"PASSED  {name}" if failure is None else f"FAILED  {name}\n        {failure}")
    n_failed = sum(failure is not None for failure in results.values())
    print(f"{len(results) - n_failed} passed, {n_failed} failed" + (" (expected files updated)" if update else ""))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", type=Path, help="case folders (default: all fixtures/case_*)")
    parser.add_argument("--update", action="store_true", help="overwrite expected_<name>.h5 with the current output")
    args = parser.parse_args()
    folders = args.folders or sorted(p for p in FIXTURES.glob("case_*") if p.is_dir())
    sys.exit(any(run_cases(folders, args.update).values()))
# test
