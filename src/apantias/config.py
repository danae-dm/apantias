from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# minor change2
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_FILE = Path("default.yaml")


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cpus: int = Field(default=4, description="Number of CPU cores")
    ram_gb: int = Field(default=8, description="RAM in GB")
    zarr_temp: Path = Field(
        default=Path("data/raw"), description="Path to zarr temp storage"
    )
    h5_archive: Path = Field(
        default=Path("data/processed"), description="Path to h5 archive"
    )


class FrameConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    rows: int = Field(default=64, description="Number of frame rows")
    cols: int = Field(default=64, description="Number of frame columns")
    nreps_eval: int = Field(
        default=200, description="Number of repetitions to be evaluated"
    )


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    frame: FrameConfig = Field(default_factory=FrameConfig)


def get_field_descriptions(model: type[BaseModel]) -> dict[str, str]:
    """Extract field descriptions from a Pydantic model."""
    descriptions: dict[str, str] = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.description:
            descriptions[field_name] = field_info.description
    return descriptions


def _print_config(config: AppConfig) -> None:
    """Pretty print configuration values with default/override indicators."""
    defaults = AppConfig()

    print("\n" + "=" * 50)
    print("Configuration Loaded:")
    print("=" * 50)
    for section_name in ["runtime", "frame"]:
        section = getattr(config, section_name)
        default_section = getattr(defaults, section_name)
        print(f"\n{section_name.upper()}:")
        for field_name, field_info in section.model_fields.items():
            value = getattr(section, field_name)
            default_value = getattr(default_section, field_name)
            desc = field_info.description or ""

            # Check if value is overridden
            is_default = value == default_value
            status = "[default]" if is_default else "[overridden]"

            print(f"  {field_name}: {value} {status}")
            if desc:
                print(f"    ({desc})")
    print("\n" + "=" * 50 + "\n")


def load_config(path: Path | None = None) -> AppConfig:
    """
    - If path is None: return defaults.
    - If path doesn't exist: write defaults to that path and return them.
    - If path exists: read & validate (merge with defaults).
    """
    if path is None:
        path = DEFAULT_CONFIG_FILE

    path = Path(path)
    if not path.exists():
        print(
            f"No file {path} found.\nA file with defaults will be created at that location."
        )
        cfg = AppConfig()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always write YAML as requested
        config_dict = cfg.model_dump()

        # Convert Path objects to strings for YAML serialization
        def path_to_str(d: dict[str, Any]) -> None:
            for k, v in d.items():
                if isinstance(v, Path):
                    d[k] = str(v)
                elif isinstance(v, dict):
                    path_to_str(v)  # type: ignore[arg-type]

        path_to_str(config_dict)

        # Build YAML with descriptions as comments
        yaml_lines: list[str] = []
        configs: list[tuple[str, type[BaseModel]]] = [
            ("runtime", RuntimeConfig),
            ("frame", FrameConfig),
        ]
        for section, config_class in configs:
            yaml_lines.append(f"{section}:")
            descriptions = get_field_descriptions(config_class)
            for key, value in config_dict[section].items():
                if key in descriptions:
                    yaml_lines.append(f"# {descriptions[key]}")
                yaml_lines.append(f"  {key}: {value}")

        path.write_text("\n".join(yaml_lines))
        _print_config(cfg)
        return cfg

    text = path.read_text()
    data: dict[str, Any] = yaml.safe_load(text) or {}

    # Validates and fills missing values with defaults
    config = AppConfig.model_validate(data)
    _print_config(config)
    return config
