#include <stdio.h>
#include <math.h>

#include "utils.h"
#include "winograd_2x2_3x3.h"

const int N = 32,
          C = 512,
          H = 28,
          W = 28,
          K = 512,
          padding = 0,
          FILTER_SIZE = 3;

int FILTER_RADIUS = FILTER_SIZE / 2,
    OUT_H = H + 2 * padding - FILTER_SIZE + 1,
    OUT_W = W + 2 * padding - FILTER_SIZE + 1;
    
float *convCPU(float *img, float *F);
float maxErr(float *A, float *B, int N);

int main()
{

    float *img = (float *)malloc(N * C * H * W * sizeof(float));
    
    initArr(img, N * C * H * W);
    // printf("\nIMG: \n");
    // printArr(img, N, C, H, W);

    float *f = (float *)malloc(K * C * 3 * 3 * sizeof(float));
    initArr(f, K * C * 3 * 3);
    // printf("\nFILTER: \n");
    // printArr(f, K, C, 3, 3);

    // float *out_actual = convCPU(img, f);
    // printf("\nOUT_ACTUAL: ");
    // printArr(out_actual, N, K, OUT_H, OUT_W);

    float *out = conv_winograd_2x2_3x3(img, N, C, H, W, f, K, padding);
    // printf("\n OUT: \n");
    // printArr(out, N, K, OUT_H, OUT_W);
    // printf("\nWINOGRAD 2x2_3x3 MAX ERR: %f", maxErr(out_actual, out, N * K * OUT_H * OUT_W));
    free(out);

    // free(out_actual);
    free(img);
    free(f);

    return 0;
}

// only works for padding 0, 1
float *convCPU(float *img, float *F)
{
    float *out = (float *)malloc(N * K * OUT_H * OUT_W * sizeof(float));
    struct _timespec64 start, end;
    long long elapsed_nanos;
    // Get the current time at the start
    _timespec64_get(&start, TIME_UTC);

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            for (int h = FILTER_RADIUS - padding; h < H - (FILTER_RADIUS - padding); ++h)
                for (int w = FILTER_RADIUS - padding; w < W - (FILTER_RADIUS - padding); ++w)
                {
                    float tmp = 0;
                    for (int c = 0; c < C; ++c)
                        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i)
                            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j)
                                if (h + i >= 0 && h + i < H && w + j >= 0 && w + j < W)
                                    tmp += img[((n * C + c) * H + h + i) * W + (w + j)] * F[((k * C + c) * FILTER_SIZE + (i + FILTER_RADIUS)) * FILTER_SIZE + (j + FILTER_RADIUS)];
                    out[((n * K + k) * OUT_H + h - (FILTER_RADIUS - padding)) * OUT_W + w - (FILTER_RADIUS - padding)] = tmp;
                }
    // Get the current time at the end
    _timespec64_get(&end, TIME_UTC);
    elapsed_nanos = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    // Print the elapsed time
    printf("Naive Conv time: %lld ms\n", elapsed_nanos / 1000000);
    return out;
}

float maxErr(float *A, float *B, int N)
{
    float maxErr = INT_MIN;
    for (int i = 0; i < W; ++i)
        maxErr = max(fabsf(B[i] - A[i]), maxErr);

    return maxErr;
}
