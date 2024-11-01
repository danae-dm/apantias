TODO:
- check documentation of functions
- write wiki
- add type hints and write nice documentation (almost done)
- consider combining analysis and analysis_funcs (rethink module naming)

new features:
- implement tracking of ignored frames (deleted because of mips or bad frames)
- check if raw_offset is really needed
- implement new event-map
- implement a gain step
- create some pictures as output

COL vs ROW Convention:

In ROOT its (col, row), but:
data is represented as (frame,row,nreps,col), so i will use (row, col) here

Steps to upload package:
https://packaging.python.org/en/latest/tutorials/packaging-projects/