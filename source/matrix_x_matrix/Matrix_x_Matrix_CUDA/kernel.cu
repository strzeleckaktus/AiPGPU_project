﻿#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define TILE_DIM 2

__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    float CValue = 0;

    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {

        if (k * TILE_DIM + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row * ACols + k * TILE_DIM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (k * TILE_DIM + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[Row * CCols + Col] = CValue;
}

void random_ints(float** matrix, size_t N, size_t M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix[i][j] = (float)(rand() % 10);
        }
    }
}

void printResults(float** A, float** B, float** C, size_t N, size_t M, size_t Csize) {
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
    printf("Matrix C:\n");
    for (int i = 0; i < Csize; i++) {
        for (int j = 0; j < Csize; j++) {
            printf("%f	", C[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int N = 3;
    int M = 5;
    int Csize = 0;
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
    random_ints(B, M, N);

    Csize = N;

    float** C = new float* [Csize];
    C[0] = new float[Csize * Csize];
    for (int i = 1; i < Csize; i++) {
        C[i] = C[0] + i * Csize;
    }

    float* cuda_A, * cuda_B, * cuda_C;

    cudaMalloc(&cuda_A, (N * M) * sizeof(float));
    cudaMalloc(&cuda_B, (N * M) * sizeof(float));
    cudaMalloc(&cuda_C, (Csize * Csize) * sizeof(float));

    cudaMemcpy(cuda_A, A[0], (N * M) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B[0], (N * M) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid((Csize * TILE_DIM - 1) / TILE_DIM, (Csize * TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    MatMul << <dimGrid, dimBlock >> > (cuda_A, cuda_B, cuda_C, N, M, M, N, Csize, Csize);

    cudaMemcpy(C[0], cuda_C, (Csize * Csize) * sizeof(float), cudaMemcpyDeviceToHost);

    printResults(A, B, C, N, M, Csize);
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    delete(A);
    delete(B);
    delete(C);

    system("pause");

    return 0;
}