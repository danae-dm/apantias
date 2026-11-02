"""Materialization strategy for intermediate (branch-point) pipeline arrays."""

# Branch-point materialization orchestrates Dask/distributed calls; both
# libraries ship incomplete stubs, so strict-mode unknown-type checks are
# relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

import logging
from pathlib import Path
from typing import cast

import dask.array as da
from dask.distributed import Client

_logger = logging.getLogger(__name__)


class Materializer:
    """Materializes branch-point arrays so they are not recomputed.

    Uses an in-memory persist when the array fits comfortably in the cluster
    RAM, otherwise writes to Zarr and reads it back.
    """

    def __init__(self, client: Client, ram_mb: int) -> None:
        # Use a quarter of the configured RAM as the per-array persist budget.
        self._budget_bytes = int(ram_mb * 1024 * 1024 * 0.25) if ram_mb > 0 else 0
        self._client = client

    def persist_or_store(self, arr: da.Array, path: Path) -> da.Array:
        if 0 < arr.nbytes < self._budget_bytes:
            _logger.info("Persist %s in memory (%.2f MB).", path.name, arr.nbytes / 1e6)
            return cast(da.Array, self._client.persist(arr))

        _logger.info("Store %s to zarr (%.2f MB).", path.name, arr.nbytes / 1e6)
        arr.to_zarr(path, overwrite=True)
        return cast(da.Array, da.from_zarr(path))
