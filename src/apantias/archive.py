"""Archive analysis outputs and clean up temporary Zarr stores."""

# h5py has no type stubs and dask arrays expose incomplete types, so relax
# strict-mode unknown-type checks in this I/O module.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py  # pyright: ignore[reportMissingTypeStubs]
import yaml
from pydantic import BaseModel

from apantias.settings import AppSettings

_logger = logging.getLogger(__name__)


def cleanup_temp(config: AppSettings) -> None:
    """Remove all temporary Zarr stores under ``config.analysis.zarr_temp``."""
    temp_root = Path(config.analysis.zarr_temp)
    if not temp_root.exists():
        return
    shutil.rmtree(temp_root)
    _logger.info("Cleaned up temporary Zarr store: %s", temp_root)


def _model_dump_recursive(obj: object) -> Any:
    """Recursively dump Pydantic models to plain dicts/lists for YAML serialization."""
    if isinstance(obj, BaseModel):
        return _model_dump_recursive(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _model_dump_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_model_dump_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_model_dump_recursive(v) for v in obj)
    return obj


def archive(config: AppSettings, outputs: Sequence[tuple[str, Any]]) -> None:
    """Archive analysis outputs and settings into a single HDF5 file.

    Parameters
    ----------
    config : AppSettings
        Full application settings; used to determine the archive path and are
        serialized into the file.
    outputs : sequence of (name, dask array) pairs
        Final pipeline outputs. Each array is computed to a NumPy array before
        writing because h5py cannot store lazy Dask arrays.
    """
    archive_path = Path(config.analysis.h5_archive)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    _logger.info("Writing analysis archive to %s", archive_path)
    with h5py.File(archive_path, "a") as f:
        for name, arr in outputs:
            if name in f:
                del f[name]
            f.create_dataset(name, data=arr.compute())

        settings_yaml = yaml.dump(_model_dump_recursive(config))
        if "settings" in f:
            del f["settings"]
        f.create_dataset("settings", data=settings_yaml)

    _logger.info("Done writing archive.")
