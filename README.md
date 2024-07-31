# Non-Fused Winograd Convolution in C
This project implements non-fused Winograd convolution: 2x2 with 3x3 filters, where the first refers to the output tile size. The implementation is done in C and includes support for padding. Stride is not supported in this version.

## Features
* <b>Winograd Convolution</b>: Implements Winograd convolution for 2x2 output tile sizes with 3x3 filters.
* <b>Padding Support</b>: Includes support for padding the input.
* <b>4 Phases of Computation</b>:
    1. <b>Filter Transform (G.g.G<sup>T</sup>)</b>
    2. <b>Input Transform (B<sup>T</sup>.d.B)</b>
    3. <b>MKL's Batched GEMM for Hadamard Product</b>
    4. <b>Inverse Transform (A<sup>T</sup>.t.A)</b>
* <b>AVX</b>: For CPUs which support AVX instructions, an AVX implementation of input transform is provided.

## Usage
To use this implementation, follow the syntax and steps outlined below:
### Syntax
```c
#include "winograd_2x2_3x3.h"

const int N = 32,
          C = 128,
          H = 112,
          W = 112,
          K = 128,
          padding = 1;

float *img = (float *)malloc(N * C * H * W * sizeof(float));
float *filter = (float *)malloc(K * C * 3 * 3 * sizeof(float));

// out [N, K, H, W]
float *out = conv_winograd_2x2_3x3(img, N, C, H, W, filter, K, padding);

free(img); free(filter); free(out);
```

## Building and Running
To build the project, ensure you have Intel OneAPI installed, at least MKL and properly configured. Use CMake to build the project.

While I've used Intel C compiler bundled with OneAPI, it can be changed as long as equivalent flags are passed. 
```bash
icx main.c -o main.exe -I "include" -march=native mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib -Qiopenmp
```
Run binary (main.exe).

## Acknowledgements
* Intel for their OneAPI base toolkit.
* <a href = "https://arxiv.org/abs/1509.09308">Fast Algorithms for Convolutional Neural Networks</a>
