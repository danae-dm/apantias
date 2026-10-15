# Changelog / Development Notes

Historical development notes for `apantias`. New entries go on top.

## Unreleased / TODO

## 2026-10-15

- refactored `standard.py` into a typed, lazy Dask pipeline:
  - added `AnalysisPaths` and `AnalysisArrays` dataclasses
  - split `StandardAnalysis` into `_load_inputs`, `_build_pipeline`, and `_save_outputs`
  - introduced `_persist_or_store` for branch-point materialization (memory or Zarr)
  - final outputs are now written together via `da.store`
- made `compute.stats` functions pure (return `da.Array` instead of writing to paths)
- made `compute.reductions.apply_pixelwise` pure and integrated it as a real stage on `data_p`
- fixed rechunk value: axis 1 is now rechunked to `n_cols` instead of `n_rows`
- fixed offset correction logic: `compute_offset_corr` now subtracts both the external
  offset and the computed median; previously the median was computed but unused and the
  external offset was passed as if it were the median
- fixed default zero-offset shape to `(n_cols, n_rows)`
- updated `tests/test_compute_stats.py` for the new pure-function API


- move functions from utils to a compute module
- edit the compute functions to use the new config
- test a fresh run (without reusing, just recalculate everything — think about adding that later on)
- think about nrepseval for the fresh run
- add a parameter on whether to keep the rawdata after the run
- think about a folder where the raw zarr data can be saved and reused
- configure dask's memory spilling
- maybe use half the processes but 2 threads per if memory problems arise
- think about different path handling (relax the .zarr requirement)
- remove the sharding stuff in data_f after the rechunk to pixels pass
- encapsulate the dask cluster creation: create it when necessary and destroy it when not
  used anymore or an exception is thrown, but keep a way to initialize it
- document everything that was done

## Operational notes

- Tested RAM requirements: minimum is 2 GB RAM per physical core.
  A chunk size of 100 MB should reliably work; lower it if RAM runs low.
  This chunk size refers to the `raw_data`, which is stored as uint16.
- Leave all parameters in the settings instance!

## 2026-10-09

- added slopes and fitting, including parameters in `settings.py`
- added a preflight module that checks everything before dask is started
- testing of the correctness of the fit is needed!

## 2026-10-03

- array layout `(n_frames, n_rows, n_reps, n_cols)` with indices
  `(frame_idx, row_idx, rep_idx, col_idx)` is now followed everywhere

## 2026-09-25

- created new `bin_to_h5` and `h5_to_zarr` functions and tested them

## 2026-09-17

- settings must now be passed to an Analysis class and are owned by it;
  the dask client is shut down after the `run()` function has finished

## 2026-09-07

- renamed `config.py` to `settings.py`
- finished the settings structure
- added module doc for how the settings should be used
