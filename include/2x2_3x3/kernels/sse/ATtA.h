#ifndef WINOGRAD_2x2_3x3_ATtA_SSE_H
#define WINOGRAD_2x2_3x3_ATtA_SSE_H

#include <xmmintrin.h>
#include <time.h>
#include <stdio.h>

float *ATtA(
    float *t, int K, int N, int TILES_H, int TILES_W, int OUT_H, int OUT_W)
{
    /*
        Input: t, [4 x 4 x K x N x TILES_H x TILES_W]
        Output: ATtA, [N x K x OUT_H x OUT_W]
        A = [
                1, 1,  1,  0
                0, 1, -1, -1
            ]
    */

    float *out = (float *)malloc(N * K * OUT_H * OUT_W * sizeof(float));
    float tmp[16], out_tmp[4];

    struct _timespec64 start, end;
    long long elapsed_nanos;
    // Get the current time at the start
    _timespec64_get(&start, TIME_UTC);
    #pragma omp parallel for collapse(2) private(tmp, out_tmp) if (N * K > 1)
    for (int n = 0; n < N; ++n)
    {
        for (int k = 0; k < K; ++k)
        {
            for (int tile_h = 0; tile_h < TILES_H; ++tile_h)
            {
                for (int tile_w = 0; tile_w < TILES_W; ++tile_w)
                {

                    // float *tmp = (float *)malloc(4 * 4 * sizeof(float));
                    for (int row_idx = 0; row_idx < 4; ++row_idx)
                    {
                        tmp[row_idx * 4 + 0] = t[((((row_idx * 4 + 0) * K + k) * N + n) * TILES_H + tile_h) * TILES_W + tile_w];
                        tmp[row_idx * 4 + 1] = t[((((row_idx * 4 + 1) * K + k) * N + n) * TILES_H + tile_h) * TILES_W + tile_w];
                        tmp[row_idx * 4 + 2] = t[((((row_idx * 4 + 2) * K + k) * N + n) * TILES_H + tile_h) * TILES_W + tile_w];
                        tmp[row_idx * 4 + 3] = t[((((row_idx * 4 + 3) * K + k) * N + n) * TILES_H + tile_h) * TILES_W + tile_w];
                    }

                    __m128 r_t0 = _mm_loadu_ps(tmp);
                    __m128 r_t1 = _mm_loadu_ps(tmp + 4);
                    __m128 r_t2 = _mm_loadu_ps(tmp + 8);
                    __m128 r_t3 = _mm_loadu_ps(tmp + 12);

                    __m128 r_ATt0 = _mm_add_ps(r_t0, _mm_add_ps(r_t1, r_t2));
                    __m128 r_ATt1 = _mm_sub_ps(r_t1, _mm_add_ps(r_t2, r_t3));

                    _mm_storeu_ps(tmp, r_ATt0);
                    _mm_storeu_ps(tmp + 4, r_ATt1);

                    // float *out_tmp = (float *)malloc(2 * 2 * sizeof(float));

                    out_tmp[0] = tmp[0] + tmp[1] + tmp[2];
                    out_tmp[1] = tmp[1] - tmp[2] - tmp[3];
                    out_tmp[2] = tmp[4 + 0] + tmp[4 + 1] + tmp[4 + 2];
                    out_tmp[3] = tmp[4 + 1] - tmp[4 + 2] - tmp[4 + 3];

                    // mapping output locations
                    int out_tile_row_idx = tile_h * 2;
                    int out_tile_col_idx = tile_w * 2;

                    for (int within_tile_row_idx = 0; within_tile_row_idx < 2; ++within_tile_row_idx)
                    {
                        for (int within_tile_col_idx = 0; within_tile_col_idx < 2; ++within_tile_col_idx)
                        {
                            int out_row_idx = within_tile_row_idx + out_tile_row_idx;
                            int out_col_idx = within_tile_col_idx + out_tile_col_idx;
                            if (out_row_idx < OUT_H && out_col_idx < OUT_W)
                                out[((n * K + k) * OUT_H + out_row_idx) * OUT_W + out_col_idx] = out_tmp[within_tile_row_idx * 2 + within_tile_col_idx];
                        }
                    }
                }
            }
        }
    }
    // Get the current time at the end
    _timespec64_get(&end, TIME_UTC);
    elapsed_nanos = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    // Print the elapsed time
    printf("Inverse Transform time: %lld us\n", elapsed_nanos / 1000);
    return out;
}

#endif