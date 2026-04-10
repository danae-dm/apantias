from dataclasses import dataclass, fields
import yaml
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when config validation fails"""

    pass


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


@dataclass
class PathConfig:
    temp_zarr: Path = Path("./zarr_temp/")
    h5_raw_data: Path = Path()

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


def load_config(config_path: Path = Path("frame_config.yaml")) -> FrameShape:
    """Load FrameShape config from YAML, creating default if not present"""

    config_path = Path(config_path)
    # Get defaults from dataclass fields
    defaults = {f.name: f.default for f in fields(FrameShape)}

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
        frame_shape = FrameShape(**config_values)
        return frame_shape
    except ConfigError as e:
        # from None prevents exception chaining
        raise ConfigError(f"Invalid frame config in {config_path}: {e}") from None
