"""Tests for ext_offset validation in the preflight check."""

# h5py/zarr ship incomplete stubs; relax the strict unknown-type checks here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
import zarr

from apantias.preflight_check import PreflightError, PreflightValidator
from apantias.settings import AnalysisSettings, AppSettings, RangeSpec

# Raw dataset shape (n_frames, n_rows, n_reps, n_cols) used in all tests.
RAW_SHAPE = (2, 4, 3, 6)


def _make_config(
    tmp_path: Path,
    ext_offset: Path | None,
    nreps_range: RangeSpec | None = None,
) -> tuple[AppSettings, Path]:
    """Create a valid raw HDF5 file and an AppSettings pointing at it."""
    h5_file = tmp_path / "raw.h5"
    data = np.zeros(RAW_SHAPE, dtype="uint16")
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("raw", data=data)

    analysis_kwargs: dict[str, Path | RangeSpec] = {"h5_file": h5_file}
    if ext_offset is not None:
        analysis_kwargs["ext_offset"] = ext_offset
    if nreps_range is not None:
        analysis_kwargs["nreps_range"] = nreps_range
    return AppSettings(analysis=AnalysisSettings(**analysis_kwargs)), h5_file  # type: ignore[arg-type]


def _write_h5_offset(tmp_path: Path, shape: tuple[int, ...], name: str = "offset") -> Path:
    ext = tmp_path / "ext_offset.h5"
    with h5py.File(ext, "w") as f:
        f.create_dataset(name, data=np.zeros(shape))
    return ext


def test_ext_offset_2d_shape_ok(tmp_path: Path) -> None:
    ext = _write_h5_offset(tmp_path, (RAW_SHAPE[1], RAW_SHAPE[3]))
    config, _ = _make_config(tmp_path, Path(f"{ext}/offset"))

    validator = PreflightValidator(config)
    enriched = validator.validate(quiet=True)

    assert validator.ctx.ext_offset_shape == (RAW_SHAPE[1], RAW_SHAPE[3])
    assert enriched.analysis.ext_offset == config.analysis.ext_offset


def test_ext_offset_3d_shape_ok(tmp_path: Path) -> None:
    ext = _write_h5_offset(tmp_path, (RAW_SHAPE[1], RAW_SHAPE[2], RAW_SHAPE[3]))
    config, _ = _make_config(tmp_path, Path(f"{ext}/offset"))

    validator = PreflightValidator(config)
    validator.validate(quiet=True)

    assert validator.ctx.ext_offset_shape == (RAW_SHAPE[1], RAW_SHAPE[2], RAW_SHAPE[3])


def test_ext_offset_3d_shape_mismatch_with_selected_reps(tmp_path: Path) -> None:
    """A 3D offset must match the reps *selected* by nreps_range, not the raw count."""
    ext = _write_h5_offset(tmp_path, (RAW_SHAPE[1], RAW_SHAPE[2], RAW_SHAPE[3]))
    # nreps_range selects only 2 of 3 reps -> effective shape (4, 2, 6).
    config, _ = _make_config(tmp_path, Path(f"{ext}/offset"), nreps_range=RangeSpec(start=0, stop=2, step=1))

    with pytest.raises(PreflightError, match="cannot be applied"):
        PreflightValidator(config).validate(quiet=True)


def test_ext_offset_wrong_shape(tmp_path: Path) -> None:
    ext = _write_h5_offset(tmp_path, (3, 3))
    config, _ = _make_config(tmp_path, Path(f"{ext}/offset"))

    with pytest.raises(PreflightError, match="cannot be applied"):
        PreflightValidator(config).validate(quiet=True)


def test_ext_offset_missing_dataset(tmp_path: Path) -> None:
    _write_h5_offset(tmp_path, (RAW_SHAPE[1], RAW_SHAPE[3]))
    config, _ = _make_config(tmp_path, Path(f"{tmp_path / 'ext_offset.h5'}/nope"))

    with pytest.raises(PreflightError, match="not found"):
        PreflightValidator(config).validate(quiet=True)


def test_ext_offset_zarr_array_ok(tmp_path: Path) -> None:
    ext = tmp_path / "ext_offset.zarr"
    zarr.open_array(
        str(ext), mode="w", shape=(RAW_SHAPE[1], RAW_SHAPE[3]), chunks=(2, 3), dtype="float64", zarr_format=3
    )
    config, _ = _make_config(tmp_path, ext)

    validator = PreflightValidator(config)
    validator.validate(quiet=True)

    assert validator.ctx.ext_offset_shape == (RAW_SHAPE[1], RAW_SHAPE[3])


def test_ext_offset_zarr_nested_array_ok(tmp_path: Path) -> None:
    ext = tmp_path / "ext_offset.zarr"
    root = zarr.open_group(str(ext), mode="w")
    root.create_array("offset", shape=(RAW_SHAPE[1], RAW_SHAPE[2], RAW_SHAPE[3]), dtype="float64")
    config, _ = _make_config(tmp_path, Path(f"{ext}/offset"))

    validator = PreflightValidator(config)
    validator.validate(quiet=True)

    assert validator.ctx.ext_offset_shape == (RAW_SHAPE[1], RAW_SHAPE[2], RAW_SHAPE[3])
