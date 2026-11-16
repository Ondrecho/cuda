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
    if (r >= N || c >= M2) return;

    int g = c / 6;
    int p = c % 6;
    int s;
    if (p == 0) s = 0;
    if (p == 1) s = 1;
    if (p == 2) s = 2;
    if (p == 3) s = 2;
    if (p == 4) s = 3;
    if (p == 5) s = 3;

    out[r * M2 + c] = in[r * M + g * 4 + s];
}

int main()
{
    double a[N * M], b[N * M2], *d_a, *d_b;
    for (int i = 0; i < N * M; i++) a[i] = i + 1;

    cudaMalloc(&d_a, sizeof(a));
    cudaMalloc(&d_b, sizeof(b));
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M2 + 15) / 16, (N + 15) / 16);
    k<<<grid, block>>>(d_a, d_b);
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
