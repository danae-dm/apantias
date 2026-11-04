"""Tests for internal helpers of the standard analysis."""

# Tests import package-internal helpers and exercise Dask/Zarr's runtime API,
# which has incomplete stubs in strict mode. Relaxed per this module's header.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from unittest.mock import Mock

import numpy as np

from apantias.pipelines import StandardPipeline


def test_load_offset_without_ext_offset() -> None:
    """Without ext_offset the applied offset is a zero array of shape (n_rows, n_cols)."""
    pipeline = StandardPipeline.__new__(StandardPipeline)
    pipeline.config = Mock()
    pipeline.config.analysis.ext_offset = None
    pipeline.config.frame.n_cols = 3
    pipeline.config.frame.n_rows = 4

    offset = pipeline._load_offset()
    assert offset.shape == (4, 3)
    np.testing.assert_array_equal(np.asarray(offset), 0)
