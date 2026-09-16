"""
Holds all the settings for apantias. The idea that a run is characterized by the settings file.
Can load all settings from a .yaml file.
if no Path is given and no default.yaml is present, a default.yaml is created
and the default values are used.

The settings are initialized in the core.init function using the set_config() function.
They can be used in all modules like this:

    from apantias.settings import get_config
    config = get_config()  # same frozen instance every call, no file read

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
from typing import Any, ClassVar

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
    # stop is exlusive! so stop=100 goes until index 99
    stop: int
    step: int = Field(default=1, ge=1)

    def __len__(self) -> int:
        return max(0, (self.stop - self.start + self.step - 1) // self.step)


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
    rows: int = Field(default=64, description="Number of frame rows")
    cols: int = Field(default=64, description="Number of frame columns")
    nreps: int = Field(default=200, description="Number of repetitions")


class AnalysisSettings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    bin_file: Path = Field(default=Path("data/bin"), description="Path to bin file")
    zarr_data: Path = Field(
        default=Path("/scratch-cbe/users/florian.heinrich/zarr_data"),
        description="Path to zarr data storage, for raw data.",
    )
    zarr_temp: Path = Field(
        default=Path("/scratch-cbe/users/florian.heinrich/zarr_temp"),
        description="Path to zarr temp storage, use fast storage options here",
    )
    h5_archive: Path = Field(default=Path("data/processed"), description="Path to h5 archive")
    ext_offset: Path | None = Field(default=None, description="Path to ext offset")
    nreps_range: RangeSpec = Field(
        default=RangeSpec(start=3, stop=200, step=1), description="Nreps range [start, stop, step]"
    )
    frames_range: RangeSpec = Field(
        default=RangeSpec(start=0, stop=100, step=1), description="Frames range [start, stop, step]"
    )


class AppSettings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    frame: FrameSettings = Field(default_factory=FrameSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)


# Process-wide frozen settings instance. None until loaded once.
_config: AppSettings | None = None


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

    for section_name, section_data in cfg.model_dump().items():
        model = type(getattr(cfg, section_name))
        descriptions = _get_field_descriptions(model)

        lines.append(f"{section_name}:")
        for key, value in section_data.items():
            if key in descriptions:
                lines.append(f"  # {descriptions[key]}")
            if isinstance(value, dict):
                # Emit nested dicts as proper YAML blocks
                lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"    {sub_key}: {sub_value}")
            else:
                lines.append(f"  {key}: {value}")

    path.write_text("\n".join(lines) + "\n")


def _validate_complete(data: dict[str, Any], path: Path) -> None:
    """Raise if the loaded config is missing any required section or field.

    Dynamic: iterates AppSettings.model_fields, so adding a new section
    to AppSettings is automatically picked up.
    """
    missing: list[str] = []

    for field_name, field_info in AppSettings.model_fields.items():
        if field_name not in data:
            # A top-level section key is entirely missing
            missing.append(field_name)
            continue
        section_data = data[field_name]
        # If the field's type is a BaseModel subclass, validate its fields too
        origin = field_info.annotation
        if origin is not None and issubclass(origin, BaseModel):
            for sub_name in origin.model_fields:
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

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} is empty or not a mapping")
    _validate_complete(data, path)
    cfg = _resolve_auto(AppSettings.model_validate(data))
    if not quiet:
        print_config(cfg)
    return cfg


def print_config(config: AppSettings) -> None:
    """Pretty print configuration values with their descriptions."""
    print("\n" + "=" * 50)
    print("Configuration Loaded:")
    print("=" * 50)

    for section_name, section in config.model_dump().items():
        print(f"\n{section_name.upper()}:")
        descriptions = _get_field_descriptions(type(getattr(config, section_name)))
        for field_name, value in section.items():
            if field_name in descriptions:
                print(f"  # {descriptions[field_name]}")
            print(f"  {field_name}: {value}")

    print("\n" + "=" * 50 + "\n")


def set_config(path: Path | str | None = None, *, overwrite: bool = False, quiet: bool = False) -> AppSettings:
    """Load the yaml config ONCE and register it as the process-wide shared instance.

    Args:
        path: Optional path to the yaml file. None uses the default config file.
        overwrite: By default raising is better than silently swapping an already
                   set config; pass True to replace it.

    Call this exactly once at startup (e.g. from your entry point). Every other
    module should then use get_config() to get the same frozen instance without
    re-reading the yaml file.
    """
    global _config

    if _config is not None and not overwrite:
        raise RuntimeError("config already set; pass overwrite=True to replace it")
    if isinstance(path, str):
        path = Path(path)
    _config = load_config(path, quiet=quiet)
    return _config


def get_config() -> AppSettings:
    """Return the shared frozen AppSettings instance.

    If set_config() was called first, this returns that exact instance.
    Throws a RuntimeError if config wasnt set
    """
    global _config
    if _config is None:
        raise RuntimeError("config not set; call set_config() first")
    return _config
