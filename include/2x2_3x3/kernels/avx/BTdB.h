#ifndef WINOGRAD_2x2_3x3_BTdB_AVX_H
#define WINOGRAD_2x2_3x3_BTdB_AVX_H

#include <immintrin.h>
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
    const int TILES_H = divUp(max(H + 2 * padding - 4, 0), 2) + 1;
    const int TILES_W = divUp(max(W + 2 * padding - 4, 0), 2) + 1;

    float *inp_transform = (float *)malloc(4 * 4 * C * N * TILES_H * TILES_W * sizeof(float));

    // number of 4x4 tiles per macro tile
    const int H_TILES_PER_MT = 2,
              W_TILES_PER_MT = 3;

    // height and width of macrotile
    const int MT_H = 4 + (H_TILES_PER_MT - 1) * 2, // 6
              MT_W = 4 + (W_TILES_PER_MT - 1) * 2; // 8

    const int MT_VS = H_TILES_PER_MT * 2, // 4 vertical stride
              MT_HS = W_TILES_PER_MT * 2; // 6 horizontal stride

    // number of such tiles per input
    const int MTILES_H = divUp(max(H + 2 * padding - MT_H, 0), MT_VS) + 1,
              MTILES_W = divUp(max(W + 2 * padding - MT_W, 0), MT_HS) + 1;

    float tmp[6 * 8] = {0}; // tmp[MT_H * MT_W]; // tmp array used below

    struct _timespec64 start, end;
    long long elapsed_nanos;
    // Get the current time at the start
    _timespec64_get(&start, TIME_UTC);

    #pragma omp parallel for collapse(2) private(tmp) if(N * C > 1)
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            const float *d_dash = d + (n * C + c) * H * W;
            float *inp_transform_dash = inp_transform + (c * N + n) * TILES_H * TILES_W;

            for (int mt_h = 0; mt_h < MTILES_H; ++mt_h)
            {
                const int h_offset = mt_h * MT_VS - padding;
                const int row_start = min(max(-h_offset, 0), MT_H);
                const int row_end = max(min(H - h_offset, MT_H), 0);
                for (int mt_w = 0; mt_w < MTILES_W; ++mt_w)
                {
                    const int w_offset = mt_w * MT_HS - padding;
                    const int col_start = min(max(-w_offset, 0), MT_W);
                    const int col_end = max(min(W - w_offset, MT_W), 0);
                    // basic way of indexing - without rowstart, rowend, colstart, colend variables  
                    // float *tmp = (float *)malloc(MT_H * MT_W * sizeof(float));
                    // for (int row_idx = 0; row_idx < MT_H; ++row_idx)
                    // {
                    //     for (int col_idx = 0; col_idx < MT_W; ++col_idx)
                    //     {
                    //         if (row_idx + h_offset >= 0 && row_idx + h_offset < H && col_idx + w_offset >= 0 && col_idx + w_offset < W)
                    //             tmp[row_idx * MT_W + col_idx] = d_dash[(row_idx + h_offset) * W + (col_idx + w_offset)];
                    //         else
                    //             tmp[row_idx * MT_W + col_idx] = 0;
                    //     }
                    // }
                    for (int row_idx = row_start; row_idx < row_end; ++row_idx)
                        for (int col_idx = col_start; col_idx < col_end; ++col_idx)
                                tmp[row_idx * MT_W + col_idx] = d_dash[(row_idx + h_offset) * W + (col_idx + w_offset)];


                    __m256 r_d[6]; // MT_H
                    for (int row_idx = 0; row_idx < MT_H; ++row_idx)
                        r_d[row_idx] = _mm256_loadu_ps(tmp + row_idx * 8);

                    // BTd
                    // vertical row khud faila le. 6 ki height k lie 2 tile
                    __m256 r_BTd[4 * 2]; // 4 * X_TILES_PER_MT
                    for (int i = 0; i < H_TILES_PER_MT; ++i)
                    {
                        r_BTd[i * 4 + 0] = _mm256_sub_ps(r_d[i * 2 + 0], r_d[i * 2 + 2]);
                        r_BTd[i * 4 + 1] = _mm256_add_ps(r_d[i * 2 + 1], r_d[i * 2 + 2]);
                        r_BTd[i * 4 + 2] = _mm256_sub_ps(r_d[i * 2 + 2], r_d[i * 2 + 1]);
                        r_BTd[i * 4 + 3] = _mm256_sub_ps(r_d[i * 2 + 1], r_d[i * 2 + 3]);
                    }

                    // transpose
                    _MM_TRANSPOSE8_PS(r_BTd[0], r_BTd[1], r_BTd[2], r_BTd[3], r_BTd[4], r_BTd[5], r_BTd[6], r_BTd[7]);

                    // mt_w * W_TILES_PER_MT + col_idx < TILES_W
                    // col_idx < TILES_W - mt_w * W_TILES_PER_MT
                    const int max_col_idx = min(TILES_W - mt_w * W_TILES_PER_MT, W_TILES_PER_MT);
                    const int max_row_idx = min(TILES_H - mt_h * H_TILES_PER_MT, H_TILES_PER_MT);
                    // 6 rows -> 8 rows. 8 cols -> 12 cols
                    for (int col_idx = 0; col_idx < max_col_idx; ++col_idx)
                    {
                        __m256 BTdB0 = _mm256_sub_ps(r_BTd[col_idx * 2 + 0], r_BTd[col_idx * 2 + 2]);
                        __m256 BTdB1 = _mm256_add_ps(r_BTd[col_idx * 2 + 1], r_BTd[col_idx * 2 + 2]);
                        __m256 BTdB2 = _mm256_sub_ps(r_BTd[col_idx * 2 + 2], r_BTd[col_idx * 2 + 1]);
                        __m256 BTdB3 = _mm256_sub_ps(r_BTd[col_idx * 2 + 1], r_BTd[col_idx * 2 + 3]);

                        // float tmp[32]; // reused the above one.
                        _mm256_storeu_ps(tmp    , BTdB0);
                        _mm256_storeu_ps(tmp + 8, BTdB1);
                        _mm256_storeu_ps(tmp + 16, BTdB2);
                        _mm256_storeu_ps(tmp + 24, BTdB3);

                        int tile_width_offset = mt_w * W_TILES_PER_MT + col_idx;
                        for (int row_idx = 0; row_idx < max_row_idx; ++row_idx)
                        {
                            int tile_height_offset = mt_h * H_TILES_PER_MT + row_idx;
                            // if (tile_height_offset < TILES_H && tile_width_offset < TILES_W) // used max_row_idx, max_col_idx to not use if.
                                for (int within_tile_row = 0; within_tile_row < 4; ++within_tile_row)
                                    for (int within_tile_col = 0; within_tile_col < 4; ++within_tile_col)
                                        inp_transform_dash[((within_tile_row * 4 + within_tile_col) * C * N * TILES_H + tile_height_offset) * TILES_W + tile_width_offset] = tmp[within_tile_col * 8 + row_idx * 4 + within_tile_row];
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
    printf("Input Transform time: %lld us\n", elapsed_nanos / 1000);

    return inp_transform;
}

#endif