TODO:
- write wiki

new features:
- rewrite filtering of "bad" Frames: MIPS and bad Frames.
    The idea is to read each frame from the data.h5 and determine if its bad or not. 
    Then, read only the good frames from the data.h5
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