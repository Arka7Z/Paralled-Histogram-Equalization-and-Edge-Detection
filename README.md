# Paralled-Histogram-Equalization-and-Edge-Detection
Perform histogram equalization and apply Sobel operator on the equalized image using MPI and libpng

## Usage: 
       mpicc -libpng -lm mpiPNG.c
       mpirung -np numProcs ./a.out fileName(in PNG format)
## Output:
       hist.pgm: Histogram equalized image.
       final.pgm: Output of applying sobel operator to histogram equalized image.
