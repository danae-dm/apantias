"""Tests for the Pipeline stage registry in apantias.io.outputs."""

from pathlib import Path

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import zarr

from apantias.io.outputs import BRANCH, DATA, Pipeline
from apantias.settings import AnalysisSettings


@pytest.fixture
def pipeline(tmp_path: Path) -> Pipeline:
    analysis = AnalysisSettings(
        zarr_data=tmp_path / "data.zarr",
        zarr_temp=tmp_path / "temp.zarr",
    )
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


def test_data_tag_moves_stage_to_zarr_data(pipeline: Pipeline) -> None:
    pipeline.declare("raw", _tiny_array(), "Raw product", tags=[DATA])
    expected = Path(pipeline.analysis.zarr_data) / "raw"
    assert pipeline.path("raw") == expected


def test_explicit_kwargs_override_tags(pipeline: Pipeline) -> None:
    # BRANCH sets is_branch_point; the explicit kwarg must win.
    pipeline.declare("s", _tiny_array(), "desc", tags=[BRANCH], is_branch_point=False)
    assert pipeline.stage("s").is_branch_point is False

    pipeline.declare("t", _tiny_array(), "desc", is_branch_point=True)
    assert pipeline.stage("t").is_branch_point is True


def test_selectors_split_branches_and_final_outputs(pipeline: Pipeline) -> None:
    pipeline.declare("median", _tiny_array(), "desc", tags=[BRANCH])
    pipeline.declare("signals", _tiny_array(), "desc", tags=[BRANCH])
    pipeline.declare("slopes", _tiny_array(), "desc")
    pipeline.declare("internal", _tiny_array(), "desc", is_archived=False)

    # Branch points keep declaration order.
    assert [s.name for s in pipeline.branch_points()] == ["median", "signals"]
    # Final outputs are archived stages that are not branch points.
    assert [s.name for s in pipeline.final_outputs()] == ["slopes"]
    # Archived includes branches; the internal stage is excluded.
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


def test_memory_budget_accounting(tmp_path: Path) -> None:
    analysis = AnalysisSettings(zarr_data=tmp_path / "d.zarr", zarr_temp=tmp_path / "t.zarr")
    pipeline = Pipeline(analysis, memory_budget_bytes=100)

    assert pipeline.can_persist_in_memory(100) is True

    pipeline.record_memory_use(70)
    assert pipeline.can_persist_in_memory(20) is True
    assert pipeline.can_persist_in_memory(40) is False  # 70 + 40 > 100


def test_memory_budget_disabled_means_unlimited(pipeline: Pipeline) -> None:
    unlimited = Pipeline(pipeline.analysis, memory_budget_bytes=None)
    assert unlimited.can_persist_in_memory(10**12) is True


def test_summary_lists_stages_and_flags(pipeline: Pipeline) -> None:
    pipeline.declare("median", _tiny_array(), "Pixel median", tags=[BRANCH])
    pipeline.declare("msd", _tiny_array(), "Mean squared deviation")

    text = pipeline.summary()
    assert "median" in text and "[branch]" in text
    assert "msd" in text and "-> archive" in text
    assert "Pixel median" in text
