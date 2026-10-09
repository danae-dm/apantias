## Testing

`src/tests/test_standard.py` is a manual regression check for `StandardAnalysis.run()`; run it before committing:

```sh
uv run python src/tests/test_standard.py                       # all src/tests/fixtures/case_* folders
uv run python src/tests/test_standard.py src/tests/fixtures/case_small_nn   # selected folders
uv run python src/tests/test_standard.py --update case_folder  # (over)write expected_<name>.h5
```

A case folder contains the raw input `<name>.h5` (dataset `raw_data`, shape `(frames, cols, nreps, rows)`) and
`expected_<name>.h5`. For each folder the analysis runs on the whole input in a temporary directory, configured
through a generated `config.yaml` that is loaded by `settings.load_config`. Then `zarr_data` must equal the input and
every `zarr_temp` array must match the dataset of the same name in the expected file (floats within 1e-10).
At the end it prints which cases passed or failed; the exit code is non-zero if any failed.

Expected files are only written with `--update`; use it for new cases or after verifying an intended result change.
Reading them elsewhere needs `import hdf5plugin`.

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

03.10.26:
DONE:
(n_frames, n_rows, n_reps, n_cols)
(frame_idx, row_idx, rep_idx, col_idx)
is now followed everywhere
