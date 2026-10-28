"""Tests for internal helpers of the standard analysis."""

# Tests import package-internal helpers and exercise Dask/Zarr's runtime API,
# which has incomplete stubs in strict mode. Relaxed per this module's header.
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from unittest.mock import Mock

import numpy as np

from apantias.standard import StandardAnalysis


def test_load_offset_without_ext_offset() -> None:
    """Without ext_offset the applied offset is a zero array of shape (n_cols, n_rows)."""
    analysis = StandardAnalysis.__new__(StandardAnalysis)
    analysis.ext_offset = None  # type: ignore[reportAttributeAccessIssue]
    analysis.config = Mock()
    analysis.config.frame.n_cols = 3
    analysis.config.frame.n_rows = 4

    offset = analysis._load_offset()
    assert offset.shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(offset), 0)
