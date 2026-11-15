#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 8
#define M 8

__global__ void k(double* in, double* out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= N || c >= M) return;

    out[r * M + c] = in[r * M + c];
}

int main()
{
    double a[N * M], b[N * M], *d_a, *d_b;
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
        for (int j = 0; j < M; j++) printf("%7g", a[i * M + j]);
        printf("\n");
    }
    printf("\nOutput:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) printf("%7g", b[i * M + j]);
        printf("\n");
    }
    return 0;
}



