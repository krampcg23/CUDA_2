# CUDA SpMV Implementation
## Author: Clayton Kramp

### Compile
To compile, type: `nvcc -o homework Clayton_Kramp_homework3.cu -O3 -D_FORCE_INLINES`

### Execute
To execute, type `./homework [file_name]`

### Notes
One of the optimizations is improving global memory access, and to do this the data and indices lines are padded with zeros to be aligned by the half-warp.  Since this prep is completed sequentially, while it improve memory access, it does take more time to run overall.
