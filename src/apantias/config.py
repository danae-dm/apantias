from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Any

import yaml

# minor change
from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cpus: int = 4
    ram_gb: int = 16
    input_path: Path = Path("data/raw")
    output_path: Path = Path("data/processed")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    n_pixels_x: int = 512
    n_pixels_y: int = 512


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    rows: int = 64
    cols: int = 64
    nreps: int = 200
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)


def load_config(path: Path | None = None) -> AppConfig:
    """
    - If path is None: return defaults.
    - If path doesn't exist: write defaults to that path and return them.
    - If path exists: read & validate (merge with defaults).
    """
    if path is None:
        return AppConfig()

    if not path.exists():
        cfg = AppConfig()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always write YAML as requested
        config_dict = cfg.model_dump()

        # Convert Path objects to strings for YAML serialization
        def path_to_str(d):
            for k, v in d.items():
                if isinstance(v, Path):
                    d[k] = str(v)
                elif isinstance(v, dict):
                    path_to_str(v)

        path_to_str(config_dict)

        path.write_text(yaml.dump(config_dict, sort_keys=False))
        return cfg

    text = path.read_text()
    data = yaml.safe_load(text) or {}

    # Validates and fills missing values with defaults
    return AppConfig.model_validate(data)
