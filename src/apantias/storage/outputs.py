"""Central declaration of pipeline outputs and their on-disk layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask.array as da

from apantias.settings import AnalysisSettings

#: Shorthand tag for outputs that should be copied into the final HDF5 archive.
#: Since ``is_archived`` defaults to False, callers must opt in with this tag.
ARCHIVE: dict[str, Any] = {"is_archived": True}


@dataclass
class Stage:
    """One named product of the analysis pipeline.

    A ``Stage`` wraps the lazy dask array together with metadata that controls
    where it is stored and whether it is archived.  The description is used
    for logging and stored as an HDF5 attribute so the archive is self-documenting.
    """

    name: str
    description: str
    array: da.Array
    path: Path
    is_archived: bool = False


@dataclass
class Pipeline:
    """Collects pipeline stages and resolves their on-disk paths from settings.

    The pipeline is the single source of truth for:

    - which arrays are produced,
    - where each array is materialised on disk,
    - which arrays end up in the final HDF5 archive.

    Note: Budget-related methods (`can_persist_in_memory`, `record_memory_use`,
    `memory_used_bytes`) are not used in the current pipeline and are provided
    for backward compatibility only.
    """

    analysis: AnalysisSettings
    _stages: dict[str, Stage] = field(default_factory=dict[str, Stage], init=False)

    def _resolve_path(self, name: str) -> Path:
        """Return the on-disk path for an output under ``zarr_temp``."""
        return Path(self.analysis.zarr_temp) / name

    def declare(
        self,
        name: str,
        array: da.Array,
        description: str,
        *,
        is_archived: bool | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> da.Array:
        """Register a new pipeline product and return its dask array unchanged.

        ``tags`` is a convenience helper: each dict is merged into the stage
        metadata, so callers can write ``tags=[ARCHIVE]`` to include a stage
        in the final HDF5 archive.

        Parameters
        ----------
        name:
            Unique identifier for the stage.  This becomes the Zarr directory
            name and the HDF5 dataset name.
        array:
            The lazy dask array produced by this stage.
        description:
            Human-readable explanation of what the array represents.
        tags:
            Optional list of shorthand tag dicts (``ARCHIVE``) to apply.
        is_archived:
            If True, the stage is copied into the final HDF5 archive.  Defaults
            to False; use ``tags=[ARCHIVE]`` to opt in.

        Note: The ``is_branch_point`` parameter is no longer used and is kept
        for backward compatibility only. All stages are treated as leaf outputs
        that are written to Zarr and then archived.
        """
        if name in self._stages:
            raise ValueError(f"Stage '{name}' has already been declared")

        # Start from defaults, apply tags, then let explicit keyword arguments
        # override everything.
        merged: dict[str, Any] = {
            "is_archived": False,
        }
        for tag in tags or []:
            merged.update(tag)
        if is_archived is not None:
            merged["is_archived"] = is_archived

        stage = Stage(
            name=name,
            description=description,
            array=array,
            path=self._resolve_path(name),
            is_archived=merged["is_archived"],
        )
        self._stages[name] = stage
        return array

    def __getitem__(self, name: str) -> da.Array:
        """Access the current dask array of a declared stage."""
        return self._stages[name].array

    def __contains__(self, name: str) -> bool:
        return name in self._stages

    def stage(self, name: str) -> Stage:
        """Return the full ``Stage`` metadata for ``name``."""
        return self._stages[name]

    def path(self, name: str) -> Path:
        """Return the on-disk Zarr path for ``name``."""
        return self._stages[name].path

    def names(self) -> list[str]:
        """Names of all declared stages in declaration order."""
        return list(self._stages.keys())

    def stages(self) -> list[Stage]:
        """All declared ``Stage`` objects in declaration order."""
        return list(self._stages.values())

    def archived(self) -> list[Stage]:
        """All stages that go into the final HDF5 archive."""
        return [s for s in self._stages.values() if s.is_archived]

    def summary(self) -> str:
        """Return a human-readable summary of declared stages."""
        lines = ["Analysis pipeline:", "-" * 50]
        for s in self._stages.values():
            archive = " -> archive" if s.is_archived else " [no archive]"
            lines.append(f"  {s.name:<15}{archive}")
            lines.append(f"    {s.description}")
        return "\n".join(lines)
