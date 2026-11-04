"""Tests for the Pipeline stage registry in apantias.storage.outputs."""

from pathlib import Path

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import zarr

from apantias.settings import AnalysisSettings
from apantias.storage.outputs import ARCHIVE, Pipeline


@pytest.fixture
def pipeline(tmp_path: Path) -> Pipeline:
    analysis = AnalysisSettings(zarr_temp=tmp_path / "temp.zarr")
    return Pipeline(analysis)


def _tiny_array() -> da.Array:
    data: npt.NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
    return da.from_array(data, chunks=(2, 2))  # type: ignore[reportReturnType, reportArgumentType]


def test_declare_registers_stage_and_returns_array(pipeline: Pipeline) -> None:
    arr = _tiny_array()
    returned = pipeline.declare("msd", arr, "Mean squared deviation")

    assert returned is arr  # declare passes the array through unchanged
    assert "msd" in pipeline
    assert pipeline["msd"] is arr
    assert pipeline.stage("msd").description == "Mean squared deviation"


def test_declare_rejects_duplicate_names(pipeline: Pipeline) -> None:
    pipeline.declare("msd", _tiny_array(), "first")
    with pytest.raises(ValueError, match="already been declared"):
        pipeline.declare("msd", _tiny_array(), "second")


def test_path_resolution_defaults_to_temp(pipeline: Pipeline) -> None:
    pipeline.declare("msd", _tiny_array(), "Mean squared deviation")
    expected = Path(pipeline.analysis.zarr_temp) / "msd"
    assert pipeline.path("msd") == expected
    assert pipeline.stage("msd").path == expected


def test_archive_tag_marks_stage_archived(pipeline: Pipeline) -> None:
    pipeline.declare("a", _tiny_array(), "desc", tags=[ARCHIVE])
    assert pipeline.stage("a").is_archived is True


def test_default_stage_is_not_archived(pipeline: Pipeline) -> None:
    pipeline.declare("a", _tiny_array(), "desc")
    assert pipeline.stage("a").is_archived is False


def test_selectors_split_archived_and_non_archived(pipeline: Pipeline) -> None:
    pipeline.declare("median", _tiny_array(), "desc", tags=[ARCHIVE])
    pipeline.declare("signals", _tiny_array(), "desc", tags=[ARCHIVE])
    pipeline.declare("slopes", _tiny_array(), "desc", tags=[ARCHIVE])
    pipeline.declare("internal", _tiny_array(), "desc")

    # Archived includes all archived stages.
    assert [s.name for s in pipeline.archived()] == ["median", "signals", "slopes"]


def test_stage_array_can_be_replaced_in_place(pipeline: Pipeline, tmp_path: Path) -> None:
    """Materialization replaces stage.array; later lookups see the new handle."""
    lazy = _tiny_array()
    pipeline.declare("msd", lazy, "desc")

    # Create the store the materialized handle reads from, as _persist_or_store would.
    store = pipeline.path("msd")
    arr: npt.NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
    zarr.open_array(  # type: ignore[reportUnknownMemberType]
        str(store), mode="w", shape=arr.shape, chunks=(2, 2), dtype=arr.dtype
    )[:] = arr
    materialized = da.from_zarr(store)  # type: ignore[reportUnknownVariableType]
    pipeline.stage("msd").array = materialized

    assert pipeline["msd"] is materialized


def test_summary_lists_stages_and_flags(pipeline: Pipeline) -> None:
    pipeline.declare("median", _tiny_array(), "Pixel median", tags=[ARCHIVE])
    pipeline.declare("msd", _tiny_array(), "Mean squared deviation", tags=[ARCHIVE])
    pipeline.declare("scratch", _tiny_array(), "Temporary result")

    text = pipeline.summary()
    assert "median" in text and "-> archive" in text
    assert "msd" in text and "-> archive" in text
    assert "scratch" in text and "[no archive]" in text
    assert "Pixel median" in text
