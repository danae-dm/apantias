# apantias

Tools for analysis of data from the DANAE experiment at HEPHY Vienna.

`apantias` ingests raw detector data, converts it to chunked `zarr` storage, and
runs a configurable, dask-based analysis pipeline (including slope fitting)
that scales across a local cluster. Configuration is centralized in a typed
settings object, and a preflight check validates the whole setup before dask
is started.

## Features

- **Data conversion**: binary raw data -> HDF5 -> zarr (`bin_to_h5`, `h5_to_zarr`)
- **Dask-backed analysis**: parallel processing with a managed local cluster
- **Central settings**: all parameters live in a single settings instance,
  passed to and owned by the `StandardAnalysis` class
- **Preflight checks**: validates configuration and resources before the run
- **Slope fitting**: with fit parameters configured in the settings

## Data layout

Arrays are stored as `(n_frames, n_rows, n_reps, n_cols)`, indexed as
`(frame_idx, row_idx, rep_idx, col_idx)`. `n_rows` is the number of readout
lines per frame (axis 1); `n_cols` is the pixels per readout line (axis 3).
Raw data is stored as `uint16`.

## Requirements

- Python 3.12
- Minimum **2 GB RAM per physical core**. A chunk size of 100 MB should
  reliably work; lower it if RAM runs low (chunk size refers to the raw
  `uint16` data).

## Installation

Install from PyPI:

```bash
uv pip install apantias
```

or with pip:

```bash
pip install apantias
```

## Quick start

```python
from pathlib import Path

from apantias import settings, standard

# Load settings from a YAML file (writes a default.yaml first if none exists)
config = settings.load_config(Path("default.yaml"))

# Hand the settings to the analysis: preflight checks run first, then a dask
# cluster is started; the client/cluster are closed again after run()
analysis = standard.StandardAnalysis(config)
analysis.run()
```

`StandardAnalysis` also accepts a path to a YAML config file
(`standard.StandardAnalysis("default.yaml")`) or no argument to use
`default.yaml` from the current directory.

All tunable parameters (chunk sizes, dask temp paths, fit thresholds, ...)
live in the frozen `AppSettings` structure — see the module docstring in
`src/apantias/settings.py`. Raw binary data can be converted with
`apantias.io.bin_to_h5` and `apantias.io.h5_to_zarr`.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync                 # install into .venv (including the package itself)
uv run pytest           # run tests (deselect slow ones: -m "not slow")
uv run ruff check .     # lint
uv run ruff format .    # format
```

## Publishing a new release

Releases are published to PyPI automatically by GitHub Actions
(`.github/workflows/python-publish.yml`) when a version tag `v*` is pushed.
Publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — no API token needed.

### Prerequisites (one-time setup)

1. **GitHub environment**: create an environment named `pypi`
   (Settings -> Environments -> New environment). No protection rules required.
2. **PyPI publisher**: on pypi.org, register the trusted publisher
   (Account settings -> Publishing -> "Add a new pending publisher"):
   - PyPI project name: `apantias`
   - Owner: `danae-dm`
   - Repository: `apantias`
   - Workflow filename: `python-publish.yml`
   - Environment: `pypi`

### Release steps

1. On the `test` branch, bump the version in `pyproject.toml` so it is unique
   (PyPI never allows re-uploading the same version):

   ```bash
   git checkout test
   git pull
   # edit version = "..." in pyproject.toml, e.g. 3.0a2 -> 3.0a3
   git commit -am "Bump version to 3.0a3"
   ```

2. Tag the release and push branch + tag. The tag (without the leading `v`)
   **must match** the version in `pyproject.toml` — the workflow verifies this
   and fails otherwise:

   ```bash
   git tag v3.0a3
   git push origin test v3.0a3
   ```

3. Watch the run at https://github.com/danae-dm/apantias/actions — it checks
   the tag/version match, builds with `uv build`, and publishes with
   `uv publish --trusted-publishing always`.

4. Verify the new version on https://pypi.org/project/apantias/

### If a run fails

- `invalid-publisher` (422) from PyPI: the trusted publisher config on
  pypi.org does not match the workflow claims (owner must be `danae-dm`,
  environment must be `pypi`). Fix it and re-run the failed job from the
  Actions page.
- Tag/version mismatch: fix the version in `pyproject.toml` or the tag, then
  delete and re-push the tag:

  ```bash
  git push origin :refs/tags/v3.0a3
  git tag -f v3.0a3
  git push origin v3.0a3
  ```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for development history and notes.

## License

MIT — see [LICENSE](LICENSE).
