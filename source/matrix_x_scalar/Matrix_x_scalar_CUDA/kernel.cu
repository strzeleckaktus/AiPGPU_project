/**
 * @mainpage Matrix Multipliaction by scalar
 * @file kernel.cu
 * @author Adrian Smoła & Kacper Godula
 * @brief All of the code pertaining to functionality of matrix by scalar multiplication
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

#define BLOCK_SIZE 32 /**< Size of CUDA blocks*/
#define MULTIPLICATOR 5 /**< Value of multiplicator*/

/**
 * @brief Kernel used to perform scalar multiplicatio
 * 
 * @param A Input Matrix A
 * @param B Output Matrix B
 * @param mul value of multiplicator
 * @param width width of matrix
 * @return __global__ 
 */
__global__ void Matrix_multiplication(float* A, float* B, float mul, int width)
{
    B[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] * mul;
}

/**
 * @brief Function to fill matrix with random integers
 * 
 * @param matrix Matrix to be modified
 * @param N Number of columns
 * @param M Number of rows
 */
void random_ints(float** matrix, size_t N, size_t M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix[i][j] = (float)(rand() % 10);
        }
    }
}

/**
 * @brief Function to print out two matrices of the same size
 * 
 * @param A Matrix A
 * @param B Matrix B
 * @param width width of a matrix 
 */
void printResults(float** A, float** B, int width) {
    printf("Matrix A:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f	", A[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f	", B[i][j]);
        }
        printf("\n");
    }
}

/**
 * @brief Main function of the file, where we declare memory and invoke kernels
 * 
 * @return int 
 */
int main() {
    int N = 4; 

    float** A = new float* [N]; 
    A[0] = new float[N * N];
    for (int i = 1; i < N; i++) {
        A[i] = A[0] + i * N;
    }
    random_ints(A, N, N);

    float** B = new float* [N];
    B[0] = new float[N * N];
    for (int i = 1; i < N; i++) {
        B[i] = B[0] + i * N;
    }

    float* cuda_A;
    float* cuda_B; 

    cudaMalloc(&cuda_A, (N * N) * sizeof(float));
    cudaMalloc(&cuda_B, (N * N) * sizeof(float));

    cudaMemcpy(cuda_A, A[0], (N * N) * sizeof(float), cudaMemcpyHostToDevice);

    Matrix_multiplication <<< ((N * N) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (cuda_A, cuda_B, MULTIPLICATOR, N);

    cudaMemcpy(B[0], cuda_B, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);

    printResults(A, B, N);

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    delete(A);
    delete(B);
    system("pause");
    return 0;

}