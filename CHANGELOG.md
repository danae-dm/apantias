# Changelog / Development Notes

Historical development notes for `apantias`. New entries go on top.

## Unreleased / TODO

<<<<<<< HEAD
## 2026-10-16

- standardized the array convention to `(n_frames, n_rows, n_reps, n_cols)` (the README
  layout) in names as well as positions: axis 1 (`n_rows`) is the number of readout
  lines per frame, axis 3 (`n_cols`) the pixels per readout line. The previous internal
  naming had `cols`/`rows` swapped relative to these meanings; the on-disk byte layout
  is unchanged, so existing HDF5/Zarr data stay valid
- fixed frame-key sentinel detection in `_parse_bin_frames`: the sentinel was looked up
  at word index `cols`, which is only correct when rows == cols; it now uses the pixel
  word count per line (this is a behaviour fix for asymmetric detectors)
- renamed HDF5 dataset attrs `column_size`/`row_size` to `n_rows`/`n_cols` (attrs were
  never read by the code)
- renamed `rechunk_to_pixels(col_batch=...)` and its internal helper to `row_batch/...`
- fixed final-output writing: `_save_outputs` passed `Path` objects to the batch
  form of `da.store`, which requires array targets (`'PosixPath' object does not
  support item assignment` on the workers); it now opens a zarr array per output
  (matching chunks, overwrite) before storing
- added asymmetric-dimension regression tests for binary frame parsing and the
  raw-to-4D mapping
- archive no longer recomputes: it takes (name, zarr path) pairs and reads the
  outputs back from their stored zarr locations
- added regression tests for batch `da.store` into zarr output arrays

=======
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f
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
<<<<<<< HEAD
=======
- fixed `apply_pixelwise`: it now truly reduces per pixel, returning shape
  `(n_cols, n_rows)` (the previous implementation passed per-row slabs of shape
  `(n_frames, n_cols, n_reps)` to the user function and crashed with
  scalar-returning functions at compute time)
- added axis constants (`FRAME`, `COL`, `REP`, `ROW`) in `compute/axis.py` and
  replaced the hardcoded axis numbers in `compute.stats`
- tightened exception handling in `preflight_check.py`:
  `except Exception` became `except (OSError, KeyError, ValueError)`
- added `write_analysis_outputs_to_h5` in `io/hdf5.py`: writes all final Zarr
  outputs into a single HDF5 archive (`AnalysisSettings.h5_archive`) as the
  last step of `_run_analysis`; large 4D arrays use frame-sized chunks with
  lzf (streamed frame by frame), small results are contiguous with gzip
- added an end-to-end integration test (`tests/test_standard_analysis_integration.py`)
  running `StandardAnalysis` on tiny synthetic data and checking every Zarr
  output and the HDF5 archive for existence, shape and value agreement
>>>>>>> 4ca6d26f462bbb59959f7acb12d920be438c4b8f


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
