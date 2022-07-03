/**
 * @mainpage Matrix Multiplication
 * @file kernel.cu
 * @author Adrian Smoła & Kacper Godula
 * @brief All of the code regarding the multiplication of matrix by another matrix
 * @version 0.1
 * @date 2022-07-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define TILE_DIM 2 /**< Size of Tiles*/

/**
 * @brief Kernel performing the calculations regarding the matrix multiplication
 * 
 * @param A Input Matrix A
 * @param B Input Matrix B
 * @param C Output Matrix
 * @param ARows Number of rows in Matrix A
 * @param ACols Number of columns in Matrix A
 * @param BRows Number of rows in Matrix B
 * @param BCols Number of columns in Matrix B
 * @param CRows Number of rows in Output Matrix
 * @param CCols Number of columns in Output Matrix
 */
__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    float CValue = 0; /**< Temporary variable*/

    int Row = blockIdx.y * TILE_DIM + threadIdx.y; /**< Current thread in y axis */
    int Col = blockIdx.x * TILE_DIM + threadIdx.x; /**< Current thread in x axis */

    __shared__ float As[TILE_DIM][TILE_DIM]; /**< Shared memory for matrix A*/
    __shared__ float Bs[TILE_DIM][TILE_DIM]; /**< Shared memory for matrix B*/

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

/**
 * @brief Function to fill a matrix with random integers
 * 
 * @param matrix Matrix to be modified
 * @param N Amount of Columns
 * @param M Amount of Rows
 */
void random_ints(float** matrix, size_t N, size_t M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix[i][j] = (float)(rand() % 10);
        }
    }
}

/**
 * @brief Function used to print all three matrices
 * 
 * @param A Matrix A
 * @param B Matrix B
 * @param C Square Matrix C
 * @param N Number of columns in matrix A and B
 * @param M Number of rows in matrix A and B
 * @param Csize size of square matrix C
 */
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

/**
 * @brief Main function where we initialize memory, and invoke kernels
 * 
 * @return int 
 */
int main() {
    int N = 3; /**< Variable to set size of input matrices*/
    int M = 5; /**< Variable to set size of input matrices*/
    int Csize = 0; /**< Size of matrix C*/
    float** A = new float* [N]; /**< Input Matrix A*/
    A[0] = new float[M * N];
    for (int i = 1; i < N; i++) {
        A[i] = A[0] + i * M;
    }
    random_ints(A, N, M);

    float** B = new float* [M]; /**< Input Matrix B*/
    B[0] = new float[M * N];
    for (int i = 1; i < M; i++) {
        B[i] = B[0] + i * N;
    }
    random_ints(B, M, N);

    Csize = N;

    float** C = new float* [Csize]; /**< Output matrix C*/
    C[0] = new float[Csize * Csize];
    for (int i = 1; i < Csize; i++) {
        C[i] = C[0] + i * Csize;
    }

    float* cuda_A; /**< Cuda memory for matrix A*/
    float * cuda_B; /**< Cuda memory for matrix B*/
    float * cuda_C; /**< Cuda memory for matrix C*/

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