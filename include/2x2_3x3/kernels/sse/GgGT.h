#ifndef WINOGRAD_2x2_3x3_GgGT_SSE_H
#define WINOGRAD_2x2_3x3_GgGT_SSE_H

#include <xmmintrin.h>
#include <time.h>
#include <stdio.h>

#include "utils.h"

float *GgGT(
    float *g, int K, int C)
{
    /*
        input -> g (filter): [K, C, 3, 3]
        output -> filter_transform : [4, 4, K, C]

        G = [
                1,    0,   0
                0.5,  0.5, 0.5
                0.5, -0.5, 0.5
                0,    0,   1
            ]
    */
    float *GgGT = (float *)malloc(4 * 4 * K * C * sizeof(float));

    float tmp[16];

    struct _timespec64 start, end;
    long long elapsed_nanos;
    // Get the current time at the start
    _timespec64_get(&start, TIME_UTC);

    #pragma omp parallel for collapse(2) private(tmp) if (K * C > 1)
    for (int k = 0; k < K; ++k)
    {
        for (int c = 0; c < C; ++c)
        {
            float *g_dash = g + (k * C + c) * 9;
            float *GgGT_dash = GgGT + (k * C + c);
            // loading g rows into register
            // g only has 3 elements, let 4th value be garbage.
            // To prevent illegal access, g's 3th row will be loaded elementwise
            __m128 r_g0 = _mm_load_ps(g_dash);     // g[0], g[1], g[2], -
            __m128 r_g1 = _mm_load_ps(g_dash + 3); // g[3], g[4], g[5], -
            __m128 r_g2 = _mm_setr_ps(g_dash[6], g_dash[7], g_dash[8], 0);

            // __m128 r_Gg0 = r_g0;
            __m128 r_tmp = _mm_add_ps(r_g0, r_g2);
            __m128 r_Gg1 = _mm_mul_ps(_mm_add_ps(r_tmp, r_g1), _mm_set1_ps(0.5));
            __m128 r_Gg2 = _mm_mul_ps(_mm_sub_ps(r_tmp, r_g1), _mm_set1_ps(0.5));
            // __m128 r_Gg3 = r_g2;

            // transpose
            _MM_TRANSPOSE4_PS(r_g0, r_Gg1, r_Gg2, r_g2); // r_g2 garbage

            // computing filter transform
            // __m128 r_GgGT0 = r_g0;
            __m128 r_tmp2 = _mm_add_ps(r_g0, r_Gg2);
            __m128 r_GgGT1 = _mm_mul_ps(_mm_add_ps(r_tmp2, r_Gg1), _mm_set1_ps(0.5));
            __m128 r_GgGT2 = _mm_mul_ps(_mm_sub_ps(r_tmp2, r_Gg1), _mm_set1_ps(0.5));
            // __m128 r_GgGT3 = r_Gg2;

            // float *tmp = (float*)_aligned_malloc(4 * 4 * sizeof(float), 16);
            _mm_storeu_ps(tmp    , r_g0);
            _mm_storeu_ps(tmp + 4, r_GgGT1);
            _mm_storeu_ps(tmp + 8, r_GgGT2);
            _mm_storeu_ps(tmp + 12, r_Gg2);

            // custom indexes hain to thoda dekh le
            for (int r = 0; r < 4; ++r)
            {
                GgGT_dash[(r * 4    ) * K * C] = tmp[    r];
                GgGT_dash[(r * 4 + 1) * K * C] = tmp[4 + r];
                GgGT_dash[(r * 4 + 2) * K * C] = tmp[8 + r];
                GgGT_dash[(r * 4 + 3) * K * C] = tmp[12 + r];
            }
        }
    }
    // Get the current time at the end
    _timespec64_get(&end, TIME_UTC);
    elapsed_nanos = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    // Print the elapsed time
    printf("Filter Transform time: %lld us\n", elapsed_nanos / 1000);
    return GgGT;
}

#endif