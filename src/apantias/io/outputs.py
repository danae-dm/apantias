"""Central declaration of pipeline outputs and their on-disk layout."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import dask.array as da

from apantias.settings import AnalysisSettings

#: Shorthand tag for branch points: arrays that are materialised so downstream
#: stages can use them.
BRANCH: dict[str, Any] = {"is_branch_point": True}

#: Shorthand tag for outputs that live in the persistent ``zarr_data`` store
#: (e.g. raw inputs).
DATA: dict[str, Any] = {"location": "data"}


@dataclass
class Stage:
    """One named product of the analysis pipeline.

    A ``Stage`` wraps the lazy dask array together with metadata that controls
    where it is stored and whether downstream stages depend on it.  The
    description is used for logging and stored as an HDF5 attribute so the
    archive is self-documenting.
    """

    name: str
    description: str
    array: da.Array
    path: Path
    is_branch_point: bool = False
    is_archived: bool = True


@dataclass
class Pipeline:
    """Collects pipeline stages and resolves their on-disk paths from settings.

    The pipeline is the single source of truth for:

    - which arrays are produced,
    - where each array is materialised on disk,
    - which arrays are branch points that downstream stages need,
    - which arrays end up in the final HDF5 archive.
    """

    analysis: AnalysisSettings
    _stages: dict[str, Stage] = field(default_factory=dict[str, Stage], init=False)

    def _resolve_path(self, name: str, location: Literal["data", "temp"]) -> Path:
        """Return the on-disk path for an output based on its storage location.

        Raw inputs live in ``zarr_data``; intermediate and final products live in
        ``zarr_temp``.  The actual roots come from ``AnalysisSettings``.
        """
        root = Path(self.analysis.zarr_data) if location == "data" else Path(self.analysis.zarr_temp)
        return root / name

    def declare(
        self,
        name: str,
        array: da.Array,
        description: str,
        *,
        location: Literal["data", "temp"] | None = None,
        is_branch_point: bool | None = None,
        is_archived: bool | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> da.Array:
        """Register a new pipeline product and return its dask array unchanged.

        ``tags`` is a convenience helper: each dict is merged into the stage
        metadata, so callers can write ``tags=[BRANCH]`` or ``tags=[DATA]``
        instead of repeating keyword arguments.  Explicit keyword arguments
        override tags.

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
            Optional list of shorthand tag dicts (``BRANCH``, ``DATA``) to
            apply.
        """
        if name in self._stages:
            raise ValueError(f"Stage '{name}' has already been declared")

        # Start from defaults, apply tags, then let explicit keyword arguments
        # override everything.
        merged: dict[str, Any] = {
            "location": "temp",
            "is_branch_point": False,
            "is_archived": True,
        }
        for tag in tags or []:
            merged.update(tag)
        if location is not None:
            merged["location"] = location
        if is_branch_point is not None:
            merged["is_branch_point"] = is_branch_point
        if is_archived is not None:
            merged["is_archived"] = is_archived

        stage = Stage(
            name=name,
            description=description,
            array=array,
            path=self._resolve_path(name, merged["location"]),
            is_branch_point=merged["is_branch_point"],
            is_archived=merged["is_archived"],
        )
        self._stages[name] = stage
        return array

    def __getitem__(self, name: str) -> da.Array:
        """Access the current dask array of a declared stage.

        After ``materialize_branches`` this returns the materialised array, so
        downstream declarations automatically use the persisted version.
        """
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

    def branch_points(self) -> list[Stage]:
        """Stages that must be materialised so downstream stages can use them."""
        return [s for s in self._stages.values() if s.is_branch_point]

    def final_outputs(self) -> list[Stage]:
        """Archived stages that are not branch points (saved at the very end)."""
        return [s for s in self._stages.values() if s.is_archived and not s.is_branch_point]

    def archived(self) -> list[Stage]:
        """All stages that go into the final HDF5 archive."""
        return [s for s in self._stages.values() if s.is_archived]

    def materialize_branches(
        self,
        materialize: Callable[[da.Array, Path], da.Array],
    ) -> None:
        """Persist every declared branch point in place.

        Because stages are looked up by name, downstream stages that access a
        branch point via ``__getitem__`` automatically see the materialised
        array.
        """
        for stage in self.branch_points():
            stage.array = materialize(stage.array, stage.path)

    def summary(self) -> str:
        """Return a human-readable summary of declared stages."""
        lines = ["Analysis pipeline:", "-" * 50]
        for s in self._stages.values():
            branch = " [branch]" if s.is_branch_point else ""
            archive = " -> archive" if s.is_archived else ""
            lines.append(f"  {s.name:<15} {branch}{archive}")
            lines.append(f"    {s.description}")
        return "\n".join(lines)
