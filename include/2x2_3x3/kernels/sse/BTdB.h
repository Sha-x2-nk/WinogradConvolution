#ifndef WINOGRAD_2x2_3x3_BTdB_SSE_H
#define WINOGRAD_2x2_3x3_BTdB_SSE_H

#include <xmmintrin.h>
#include <stdio.h>
#include <time.h>

#include "utils.h"

float *BTdB(
    float *d, int N, int C, int H, int W, int padding)
{
    /*
        Input: d = img [N, C, H, W]
        Output: input_transform [4, 4, C, N, TILES_H, TILES_W]

        BT = [
                1, 0, -1, 0
                0, 1, 1, 0
                0, -1, 1, 0
                0, 1, 0, -1
             ]
    */
    const int TILES_H = divUp(H + 2 * padding - 4, 2) + 1;
    const int TILES_W = divUp(W + 2 * padding - 4, 2) + 1;

    float *inp_transform = (float *)malloc(4 * 4 * C * N * TILES_H * TILES_W * sizeof(float));

    float tmp[16];

    struct _timespec64 start, end;
    long long elapsed_nanos;
    // Get the current time at the start
    _timespec64_get(&start, TIME_UTC);
    #pragma omp parallel for collapse(2) private(tmp) if (N * C > 1)
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            float *d_dash = d + (n * C + c) * H * W;
            float *inp_transform_dash = inp_transform + (c * N + n) * TILES_H * TILES_W;

            for (int tile_h = 0; tile_h < TILES_H; ++tile_h)
            {
                int h_offset = tile_h * 2 - padding;

                for (int tile_w = 0; tile_w < TILES_W; ++tile_w)
                {
                    int w_offset = tile_w * 2 - padding;

                    // loading tile with padding
                    // float *tmp = (float *)malloc(4 * 4 * sizeof(float));
                    for (int row_idx = 0; row_idx < 4; ++row_idx)
                    {
                        for (int col_idx = 0; col_idx < 4; ++col_idx)
                            if (h_offset + row_idx >= 0 && h_offset + row_idx < H && w_offset + col_idx >= 0 && w_offset + col_idx < W)
                                tmp[row_idx * 4 + col_idx] = d_dash[(h_offset + row_idx) * W + (w_offset + col_idx)];
                            else
                                tmp[row_idx * 4 + col_idx] = 0;
                    }

                    __m128 r_d0 = _mm_loadu_ps(tmp);
                    __m128 r_d1 = _mm_loadu_ps(tmp + 4);
                    __m128 r_d2 = _mm_loadu_ps(tmp + 8);
                    __m128 r_d3 = _mm_loadu_ps(tmp + 12);

                    // BTd
                    __m128 r_BTd0 = _mm_sub_ps(r_d0, r_d2);
                    __m128 r_BTd1 = _mm_add_ps(r_d1, r_d2);
                    __m128 r_BTd2 = _mm_sub_ps(r_d2, r_d1);
                    __m128 r_BTd3 = _mm_sub_ps(r_d1, r_d3);

                    _MM_TRANSPOSE4_PS(r_BTd0, r_BTd1, r_BTd2, r_BTd3);

                    // BTdB
                    __m128 r_BTdB0 = _mm_sub_ps(r_BTd0, r_BTd2);
                    __m128 r_BTdB1 = _mm_add_ps(r_BTd1, r_BTd2);
                    __m128 r_BTdB2 = _mm_sub_ps(r_BTd2, r_BTd1);
                    __m128 r_BTdB3 = _mm_sub_ps(r_BTd1, r_BTd3);

                    _mm_storeu_ps(tmp, r_BTdB0);
                    _mm_storeu_ps(tmp + 4, r_BTdB1);
                    _mm_storeu_ps(tmp + 8, r_BTdB2);
                    _mm_storeu_ps(tmp + 12, r_BTdB3);

                    // storing to RAM
                    for (int r = 0; r < 4; ++r)
                    {
                        inp_transform_dash[((r * 4 + 0) * C * N * TILES_H + tile_h) * TILES_W + tile_w] = tmp[0 + r];
                        inp_transform_dash[((r * 4 + 1) * C * N * TILES_H + tile_h) * TILES_W + tile_w] = tmp[4 + r];
                        inp_transform_dash[((r * 4 + 2) * C * N * TILES_H + tile_h) * TILES_W + tile_w] = tmp[8 + r];
                        inp_transform_dash[((r * 4 + 3) * C * N * TILES_H + tile_h) * TILES_W + tile_w] = tmp[12 + r];
                    }
                }
            }
        }
    }
    // Get the current time at the end
    _timespec64_get(&end, TIME_UTC);
    elapsed_nanos = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    // Print the elapsed time
    printf("Input Transform time: %lld us\n", elapsed_nanos / 1000);
    return inp_transform;
}

#endif