#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define TILE_DIM 3

__global__ void Matrix_transpose(float* A, float* B, int A_rows, int A_cols) {

    float CValue = 0;

    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];

    if (Row < A_rows && Col < A_cols)
        As[threadIdx.x][threadIdx.y] = A[Row * A_cols + Col];

    //B[Row + Col*A_cols] = As[threadIdx.x][threadIdx.y];
    if (Row < A_rows && Col < A_cols)
        B[Row + Col * A_cols] = As[threadIdx.x][threadIdx.y];
}

void random_ints(float** matrix, size_t N, size_t M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix[i][j] = (float)i*N + j;
        }
    }
}

void printResults(float** A, float** B, size_t N, size_t M) {
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f	", A[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f	", B[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int N = 5;
    int M = 5;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float** A = new float* [N];
    A[0] = new float[M * N];
    for (int i = 1; i < N; i++) {
        A[i] = A[0] + i * M;
    }
    random_ints(A, N, M);

    float** B = new float* [M];
    B[0] = new float[M * N];
    for (int i = 1; i < M; i++) {
        B[i] = B[0] + i * N;
    }

    float* cuda_A, * cuda_B;

    cudaMalloc(&cuda_A, (N * M) * sizeof(float));
    cudaMalloc(&cuda_B, (N * M) * sizeof(float));

    cudaMemcpy(cuda_A, A[0], (N * M) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid((N * TILE_DIM - 1) / TILE_DIM, (N * TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    cudaEventRecord(start);

    Matrix_transpose << <dimGrid, dimBlock >> > (cuda_A, cuda_B, N, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(B[0], cuda_B, (N * M) * sizeof(float), cudaMemcpyDeviceToHost);

    printResults(A, B, N, M);
    //printResults(A, B, N, M);
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    delete(A);
    delete(B);

    system("pause");

    return 0;
}