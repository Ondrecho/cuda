#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 4
#define M 8
#define M2 (M / 4 * 6)

__global__ void k(double* in, double* out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= N || c >= M) return;

    double v = in[r * M + c];
    int g = c / 4;
    int p = c % 4;

    if (p == 0) {
        out[r * M2 + g * 6 + 0] = v;
    }
    if (p == 1) {
        out[r * M2 + g * 6 + 1] = v;
    }
    if (p == 2) {
        out[r * M2 + g * 6 + 2] = v;
        out[r * M2 + g * 6 + 3] = v;
    }
    if (p == 3) {
        out[r * M2 + g * 6 + 4] = v;
        out[r * M2 + g * 6 + 5] = v;
    }
}

int main()
{
    double a[N * M], b[N * M2], *d_a, *d_b;
    for (int i = 0; i < N * M; i++) a[i] = i + 1;

    cudaMalloc(&d_a, sizeof(a));
    cudaMalloc(&d_b, sizeof(b));
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);

    k<<<dim3(1, 1), dim3(M, N)>>>(d_a, d_b);
    cudaDeviceSynchronize();

    cudaMemcpy(b, d_b, sizeof(b), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);

    printf("Input:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) printf("%5g", a[i * M + j]);
        printf("\n");
    }
    printf("\nOutput:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M2; j++) printf("%5g", b[i * M2 + j]);
        printf("\n");
    }
    return 0;
}
