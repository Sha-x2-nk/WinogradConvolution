#ifndef WINOGRAD_2x2_3x3_H
#define WINOGRAD_2x2_3x3_H

#include "2x2_3x3/kernels/sse/GgGT.h"      // filter transform
// #include "2x2_3x3/kernels/sse/BTdB.h"      // input transform - SSE
#include "2x2_3x3/kernels/avx/BTdB.h"      // input_transform - AVX
#include "2x2_3x3/kernels/sse/ATtA.h"      // inverse transform

#include <mkl.h>

#include <stdio.h>
#include <stdlib.h> // max, min
#include <time.h>

/*
    filter_transform = [4, 4, K, C]
    input_transform = [4, 4, C, N, th, tw]
    hadamard = [4, 4, K, N, th, tw]
    inverse transform = [4, 4, K, N, th, tw] -> [2, 2, th, tw] (per K, N) -> [N, K, H, W]
*/

float *conv_winograd_2x2_3x3(float *img, int N, int C, int H, int W, float *f, int K, int padding)
{

    float *filter_transform = GgGT(f, K, C);

    float *inp_transform = BTdB(img, N, C, H, W, padding);
    // printf("\nINP TRANSFORM: ");
    // printArr(inp_transform, 4, 4, , 1);
    
    const int TILES_H = divUp(max(H + 2 * padding - 4, 0), 2) + 1;
    const int TILES_W = divUp(max(W + 2 * padding - 4, 0), 2) + 1;
    const int OUT_H = (H + 2 * padding - 3 + 1);
    const int OUT_W = (W + 2 * padding - 3 + 1);

    float *M = (float *)malloc(4 * 4 * K * N * TILES_H * TILES_W * sizeof(float));
    {
        struct _timespec64 start, end;
        long long elapsed_nanos;
        // Get the current time at the start
        _timespec64_get(&start, TIME_UTC);
        cblas_sgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                  K, N * TILES_H * TILES_W, C,
                                  1.0f,
                                  filter_transform, C, K * C,
                                  inp_transform, N * TILES_H * TILES_W, C * N * TILES_H * TILES_W,
                                  0.0f,
                                  M, N * TILES_H * TILES_W, K * N * TILES_H * TILES_W,
                                  4 * 4);
        // Get the current time at the end
        _timespec64_get(&end, TIME_UTC);
        elapsed_nanos = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
        // Print the elapsed time
        printf("Hadamard time: %lld us\n", elapsed_nanos / 1000);
    }

    free(filter_transform);
    free(inp_transform);

    float *out = ATtA(M, K, N, TILES_H, TILES_W, OUT_H, OUT_W);

    free(M);
    return out;
}

#endif