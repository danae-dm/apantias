## Testing

`src/tests/test_standard.py` is a regression test for `StandardAnalysis(config_path).run()` (`src/apantias/standard.py`).

- **Fixtures** (`src/tests/fixtures/`, gitignored; without it the test is skipped): every `case_<name>/` directory is one
  case, test ID `<name>`. It must contain the raw input `<name>.h5` (dataset `raw_data`, shape
  `(frames, cols, nreps, rows)`) and the golden output `expected_<name>.h5`. Other `.h5` files make collection fail.
- **Config**: there are no YAML files in the fixtures. For each case the test builds an `AppSettings`, with frame
  layout and full `nreps_range`/`frames_range` taken from the input shape and `cpus: 4`. It writes this as
  `config.yaml` to pytest's `tmp_path` and passes the path to `StandardAnalysis`, which loads it through the
  production `settings.load_config` (completeness check + pydantic validation).
- **Paths**: all paths are absolute. `h5_file` points read-only at the fixture input; `dask_temp`, `zarr_data`,
  `zarr_temp` and `h5_archive` point into `tmp_path`, and the test runs with `tmp_path` as cwd. The test fails if
  anything in the case directory changes or if unexpected entries appear in `tmp_path`.
- **Checks**: `run()` returns `None`. `zarr_data/{frame_chunked,pixel_chunked}` must equal the input exactly. Every
  array in `zarr_temp` (median, offset_corr, common_modes, signals, slopes, msd, signals_mean, test_array) must match
  the dataset with the same name in `expected_<name>.h5`: same names, shapes and dtypes; floats within rtol=atol=1e-10.
  The golden file also stores the result-relevant settings; if they no longer match, the test asks for regeneration.
  Reading golden files outside the test needs `import hdf5plugin` (Blosc compression).

```sh
uv run pytest -vv                                   # all tests (~30 s per case)
uv run pytest -vv "src/tests/test_standard.py::test_run_regression[small_nn]"   # one case
uv run pytest --lf -vv                              # rerun last failures
uv run pytest -x --pdb                              # stop at first failure in the debugger
```

**Golden files are never updated automatically.** To add a case, create `src/tests/fixtures/case_<name>/<name>.h5`
and run `uv run pytest --update-golden -vv "src/tests/test_standard.py::test_run_regression[<name>]"`. Pass
`--update-golden` only after reviewing that the current output is correct, because it overwrites `expected_<name>.h5`.

historical notes:

Tested RAM requirements:
Minimum is 2GB RAM per physical core.
A chunk size of 100MB should reliably work. Must be lowered if RAM runs low.
This chunk size is for the raw_data which is stored in uint16

leave all parameters in the settings instance!

TODO:
move functions from utils to a compute module
edit the compute functions to use the new config
test a fresh run (without reusing, just recalculate everything think about adding that later on)
think about nrepseval for the fresh run
add a parameter on whether to keep the rawdata after the run
think about a folder where the raw zarr data can be saved and reused.
configure dasks memory spilling
maybe use half the processes but 2 threads per if memory problems arise
think about different path handling (relax the .zarr requirement)
remove the sharding stuff in data_f after the rechunk to pixels pass
encapsulate the dask cluster creation. when running it its not necessary to watch the graph, so no need to keep the cluster alive. create it when necessary and destroy it when not used anymore or a exception is thrown. but keep a way to initialize it
DOCUMENT everything that was done

07.09.26:
DONE:
renamed config.py to settings.py
finished the settings structure
added module doc for how the settings should be used


17.09.26:
DONE:
settings must now be passed to a Analysis Class and is owned by it, client is shut down after the run() function of the Analysis class has finished

25.09.26:
DONE:
created new bin_to_h5 and h5_to_zarr functions and tested them
