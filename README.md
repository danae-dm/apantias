# APANTIAS

Tools for analysis of data from the DANAE experiment on HEPHY.

## Installation

```bash
pip install apantias
```

## Documentation

The full documentation is in the [wiki](https://github.com/danae-dm/apantias/wiki).

## Architecture

APANTIAS uses a **modular** design:

- `PreflightValidator` and `runtime.cluster.init()` are public utilities that stay **outside** the pipeline.
- A user loads one `settings.yaml`, validates it, starts one Dask cluster, and can then run one or more pipelines using the same validated settings and the same client.
- `StandardPipeline` accepts and uses a Dask client.

For the quick start and detailed architecture documentation, see the wiki.
