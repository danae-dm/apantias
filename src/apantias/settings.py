"""
Holds all the settings for apantias. The idea that a run is characterized by the settings file.
Can load all settings from a .yaml file.
if no Path is given and no default.yaml is present, a default.yaml is created
and the default values are used.

Each setting is owned by a Analysis class instance

Settings should be grouped in a structured way:
    - runtime
    - frame

How to add new settings class:
    - Define the class (eg. LoggingSettings)
    - Add it to the AppSettings class

How to add new parameter to existing class:
    - just add it to the class
    - dont forget a default and description

Add validation here later, so only sensible settings will be loaded before a run. (for example RAM/Core ratio)
"""

from pathlib import Path
from typing import Any, ClassVar, cast, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import get_resources

DEFAULT_CONFIG_FILE = Path("default.yaml")
# teach pyYAML how to dump Path Objects
yaml.add_representer(Path, lambda dumper, data: dumper.represent_str(str(data)))


class RangeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    # start is inclusive
    start: int
    # stop is exclusive! -1 means up to the end of the dataset dimension
    stop: int
    step: int = Field(default=1, ge=1)

    def resolve_stop(self, size: int) -> int:
        """Resolve stop: -1 becomes dataset size."""
        return size if self.stop == -1 else self.stop

    def indices(self, size: int) -> list[int]:
        """Generate concrete indices resolved against dataset dimension size."""
        return list(range(self.start, self.resolve_stop(size), self.step))

    def count(self, size: int) -> int:
        """Return the number of elements selected when resolved against size."""
        return len(self.indices(size))

    def __len__(self) -> int:
        if self.stop == -1:
            raise ValueError(
                "Cannot compute len() of RangeSpec with stop=-1 without dataset size. "
                "Use .count(size) or .indices(size)."
            )
        return max(0, (self.stop - self.start + self.step - 1) // self.step)

    def __iter__(self) -> "RangeSpec":  # type: ignore[return-value]
        """Iterate to produce indices via Python's range()."""
        self._iterator = iter(range(self.start, self.stop, self.step))
        return self

    def __next__(self) -> int:
        return next(self._iterator)


class RuntimeSettings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    cpus: int = Field(default=0, description="Number of CPU cores, 0 is auto-detect")
    ram_mb: int = Field(default=0, description="RAM in MB, 0 is auto-detect")
    dask_temp: Path = Field(
        default=Path("/scratch-cbe/users/florian.heinrich/dask_temp"),
        description="Path to dasks temp storage, use fast storage options here",
    )


class FrameSettings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    # Inferred from HDF5 at runtime (hidden from YAML)
    n_frames: int = Field(default=0, exclude=True, description="Total number of frames (inferred from HDF5)")
    n_rows: int = Field(default=64, exclude=True, description="Number of frame rows (inferred from HDF5)")
    n_cols: int = Field(default=64, exclude=True, description="Number of frame columns (inferred from HDF5)")
    n_reps: int = Field(default=200, exclude=True, description="Number of repetitions (inferred from HDF5)")

    # User-visible configuration (kept in YAML)
    target_chunk_mb: int = Field(
        default=100,
        ge=1,
        description="Target chunk size in MB for frame writes.",
    )

    # Backwards-compatible aliases used by the binary/Zarr I/O pipeline.
    # These are simple property aliases so the public API can use the shorter
    # names (rows/cols/nreps) while the YAML/serialised model keeps the more
    # explicit n_* names.
    @property
    def rows(self) -> int:
        return self.n_rows

    @property
    def cols(self) -> int:
        return self.n_cols

    @property
    def nreps(self) -> int:
        return self.n_reps

    @property
    def target_chunk_bytes(self) -> int:
        return self.target_chunk_mb * 1024 * 1024


class AnalysisSettings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    h5_file: Path = Field(default=Path("data.h5"), description="Path to h5 file with raw data")
    zarr_data: Path = Field(
        default=Path("data.zarr"),
        description="Path to zarr data storage, for raw data.",
    )
    zarr_temp: Path = Field(
        default=Path("temp.zarr"),
        description="Path to zarr temp storage, use fast storage options here",
    )
    h5_archive: Path = Field(default=Path("archive.h5"), description="Path to h5 archive")
    ext_offset: Path | None = Field(default=None, description="Path to ext offset")
    nreps_range: RangeSpec = Field(
        default=RangeSpec(start=0, stop=-1, step=1), description="Nreps range [start, stop, step] default stop=-1"
    )
    frames_range: RangeSpec = Field(
        default=RangeSpec(start=0, stop=-1, step=1), description="Frames range [start, stop, step] default stop=-1"
    )
    slope_threshold: int = Field(default=3, description="Threshold for the Slope Filter.")
    second_offset: bool = Field(default=True, description="Shifts the signals according to the pixel hist fit.")


class AppSettings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    frame: FrameSettings = Field(default_factory=FrameSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)


def _get_field_descriptions(model: type[BaseModel]) -> dict[str, str]:
    """Extract field descriptions from a Pydantic model."""
    descriptions: dict[str, str] = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.description:
            descriptions[field_name] = field_info.description
    return descriptions


def _dump_config(cfg: AppSettings, path: Path) -> None:
    """Write default config to YAML with comments from field descriptions."""
    lines: list[str] = []

    dumped: dict[str, Any] = cfg.model_dump()
    for section_name, section_data in dumped.items():
        section_value = getattr(cfg, section_name)
        model = cast(type[BaseModel], type(section_value))
        descriptions = _get_field_descriptions(model)

        lines.append(f"{section_name}:")
        section_dict = cast(dict[str, Any], section_data)
        for key, value in section_dict.items():
            if key in descriptions:
                lines.append(f"  # {descriptions[key]}")
            if isinstance(value, dict):
                # Emit nested dicts as proper YAML blocks
                lines.append(f"  {key}:")
                nested = cast(dict[str, Any], value)
                for sub_key, sub_value in nested.items():
                    lines.append(f"    {sub_key}: {sub_value}")
            else:
                lines.append(f"  {key}: {value}")

    path.write_text("\n".join(lines) + "\n")


def _validate_complete(data: dict[str, Any], path: Path) -> None:
    """Raise if the loaded config is missing any required section or field.

    Fields marked with `exclude=True` in their Pydantic definition are internal
    runtime parameters (e.g. inferred from HDF5) and are not expected in YAML.
    All other fields must be explicitly present in the YAML file.
    """
    missing: list[str] = []

    for field_name, field_info in AppSettings.model_fields.items():
        if field_info.exclude:
            continue

        if field_name not in data:
            missing.append(field_name)
            continue

        section_data = cast(dict[str, Any], data[field_name])
        origin = get_origin(field_info.annotation)
        if isinstance(origin, type) and issubclass(origin, BaseModel):
            for sub_name, sub_info in origin.model_fields.items():
                # Skip internal fields marked with exclude=True
                if sub_info.exclude:
                    continue
                # All other fields MUST be present in the YAML
                if sub_name not in section_data:
                    missing.append(f"{field_name}.{sub_name}")

    if missing:
        raise ValueError(f"{path}: missing required field(s): {', '.join(missing)}")


def _resolve_auto(config: AppSettings) -> AppSettings:
    """Fill in cpus/ram_mb when set to 0 (auto-detect)."""
    runtime = config.runtime
    cpus, ram_mb = get_resources.get_resources()

    updates: dict[str, Any] = {}
    if runtime.cpus == 0:
        updates["cpus"] = cpus
    if runtime.ram_mb == 0:
        updates["ram_mb"] = ram_mb

    if not updates:
        return config
    return config.model_copy(update={"runtime": runtime.model_copy(update=updates)})


def load_config(path: Path | None = None, *, quiet: bool = False) -> AppSettings:
    """
    - If path is None or file missing: write defaults to default.yaml
      at the given path (or working directory) and return them.
    - If path exists: read and validate it.
    Will raise if the loaded config is missing any required section or field.
    """
    path = Path(path) if path else DEFAULT_CONFIG_FILE
    if not path.exists():
        print(f"No file {path} found.\nA file with defaults will be created at that location.")
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = AppSettings()
        _dump_config(cfg, path)
        cfg = _resolve_auto(cfg)
        if not quiet:
            print_config(cfg)
        return cfg

    data: Any = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} is empty or not a mapping")
    _validate_complete(cast(dict[str, Any], data), path)
    cfg = _resolve_auto(AppSettings.model_validate(data))
    if not quiet:
        print_config(cfg)
    return cfg


def print_config(config: AppSettings) -> None:
    """Pretty print configuration values with their descriptions."""
    print("\n" + "=" * 50)
    print("Configuration Loaded:")
    print("=" * 50)

    dumped: dict[str, Any] = config.model_dump()
    for section_name, section in dumped.items():
        print(f"\n{section_name.upper()}:")
        section_value = getattr(config, section_name)
        descriptions = _get_field_descriptions(cast(type[BaseModel], type(section_value)))
        section_dict = cast(dict[str, Any], section)
        for field_name, value in section_dict.items():
            if field_name in descriptions:
                print(f"  # {descriptions[field_name]}")
            print(f"  {field_name}: {value}")

    print("\n" + "=" * 50 + "\n")
