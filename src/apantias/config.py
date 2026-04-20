from dataclasses import dataclass, fields
import yaml
from pathlib import Path
from typing import Any, Type, TypeVar, Optional

# TODO: make it read from a single yaml file!
T = TypeVar("T")


class ConfigError(Exception):
    """Raised when config validation fails"""

    pass


def _load_yaml_config(config_class: Type[T], config_path: Path) -> T:
    """
    Generic helper to load a config dataclass from YAML.
    Creates default file if it doesn't exist.
    """
    config_path = Path(config_path)  # Reassign to narrow the type
    defaults = {f.name: f.default for f in fields(config_class)}  # type: ignore

    # Create default file if it doesn't exist
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(defaults, f, default_flow_style=False)
        print(f"Created default config file: {config_path}")

    # Load config from YAML
    with open(config_path, "r") as f:
        config_data: dict[str, Any] = yaml.safe_load(f) or {}

    try:
        # Merge YAML data with defaults (YAML values override defaults)
        config_values: dict[str, Any] = {**defaults, **config_data}
        return config_class(**config_values)
    except Exception as e:
        raise ConfigError(
            f"Invalid config for {config_class.__name__} in {config_path}: {e}"
        ) from None


@dataclass
class FrameShape:
    """
    A rectangular sensor:
    Indices start in the top left corner.
    columns from left to right
    rows from top to bottom
    rows are read repeatedly, nrep times
    """

    columns: int = 64
    rows: int = 64
    nreps: int = 200

    def __post_init__(self):
        if self.columns < 1 or self.columns > 256:
            raise ConfigError("columns must be an integer between 1-256")
        if self.rows < 1 or self.rows > 256:
            raise ConfigError("columns must be an integer between 1-256")
        if self.rows < 1 or self.rows > 2000:
            raise ConfigError("columns must be an integer between 1-256")

    def __str__(self):
        return (
            f"FrameShape(columns={self.columns}, rows={self.rows}, nreps={self.nreps})"
        )

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "FrameShape":
        """Load FrameShape config from YAML, creating default if not present"""
        if config_path is None:
            config_path = Path("frame_config.yaml")
        return _load_yaml_config(FrameShape, config_path)


@dataclass
class PathConfig:
    temp_zarr: Path = Path("./zarr_temp/")
    h5_raw_data: Path = Path("./raw_data.h5")

    def __post_init__(self):
        self.temp_zarr = Path(self.temp_zarr)
        self.h5_raw_data = Path(self.h5_raw_data)

        # Check if temp_zarr exists
        if not self.temp_zarr.exists():
            # create the directory if not
            self.temp_zarr.mkdir(parents=True, exist_ok=True)

        # Check if h5_raw_data exists (skip if empty)
        if not self.h5_raw_data.exists():
            raise ConfigError(f"h5_raw_data path does not exist: {self.h5_raw_data}")

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "PathConfig":
        """Load PathConfig from YAML, creating default if not present"""
        if config_path is None:
            config_path = Path("path_config.yaml")
        return _load_yaml_config(PathConfig, config_path)


@dataclass
class AppConfig:
    """
    Main application config containing all configuration sections.
    Load with: config = AppConfig.load()
    """

    config_path = Path("./config.yaml")
    frame_shape: FrameShape
    paths: PathConfig

    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> "AppConfig":
        """Load all configs from YAML files in base_dir"""
        if base_dir is None:
            base_dir = Path(".")
        else:
            base_dir = Path(base_dir)

        frame_shape = FrameShape.load(base_dir / "frame_config.yaml")
        paths = PathConfig.load(base_dir / "path_config.yaml")
        return cls(frame_shape=frame_shape, paths=paths)
