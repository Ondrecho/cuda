#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 8      // строк (кратно 4)
#define M 1024   // столбцов

// Окно 4x1 = 4 строки в одном столбце, все 4 элемента одинаковые.
// Окна нумеруются слева направо: 1, 2, 3 ... 1024, потом 1025 ... 2048
__global__ void k(double* out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // столбец
    int r = blockIdx.y * blockDim.y + threadIdx.y;   // строка
    if (r >= N || c >= M) return;

    int wr = r / 4;                    // номер строки окон
    out[r * M + c] = wr * M + c + 1;
}

int main()
{
    static double b[N * M];
    double* d_b;

    cudaMalloc(&d_b, sizeof(b));

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    k<<<grid, block>>>(d_b);
    cudaDeviceSynchronize();

    cudaMemcpy(b, d_b, sizeof(b), cudaMemcpyDeviceToHost);
    cudaFree(d_b);

    // Печать как в задании: первые 3 столбца ... последний
    for (int i = 0; i < N; i++) {
        printf("%6g %6g %6g ... %6g\n",
               b[i * M + 0], b[i * M + 1], b[i * M + 2], b[i * M + M - 1]);
    }
    return 0;
}
