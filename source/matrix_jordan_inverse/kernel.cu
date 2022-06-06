#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

using namespace std;

__global__ void gaussjordan(float* A, float* I, int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float P;

    if (x < n && y < n)
        if (x > i) { // this limits operation to rows below the pivot point
            P = A[x * n + i] / A[i * n + i];
            I[x * n + y] -= I[i * n + y] * P;  // apply for every row member
            if (y >= i) { //limits  to row members to the right of the pivot
                A[x * n + y] -= A[i * n + y] * P;  // apply only to members right of pivot
            }
        }
}


__global__ void dev(float* d_A, float* dI, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < h && y < h)
        if (d_A[x * h + x] != 0) {
            dI[x * h + y] /= d_A[x * h + x];
            d_A[x * h + y] /= d_A[x * h + x];
        }
    __syncthreads();

}

void savetofile(float* A, string s, int n, int h)
{
    std::ofstream plik;
    plik.open(s);

    for (int j = 0; j < h; j++) {
        for (int i = 0; i < h; i++) {
            plik << A[j * n + i] << "\t";
        }
        plik << endl;
    }
    plik.close();
}

void random_floats(float* vect, int N) {
    for (int i = 0; i < N; i++) {
        vect[i] =static_cast<float>(rand())/(static_cast<float>(RAND_MAX/50));
    }
}


int main()
{
    int n = 16;
    // creating input
    float* iL = new float[n * n];
    float* L = new float[n * n];
    random_floats(L, n * n);
    savetofile(L, "Input_matrix.txt", n, n);

    cout << "inv\n";
    float* d_A, * I, * dI;
    float time;
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ddsize = n * n * sizeof(float);

    dim3 threadsPerBlock(n / 16, n / 16);
    dim3 numBlocks(16, 16);
    // memory allocation    
    err = cudaMalloc((void**)&d_A, ddsize);   if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    err = cudaMalloc((void**)&dI, ddsize);   if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    I = new float[n * n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) I[i * n + i] = 1.0;
            else I[i * n + j] = 0.0;
        }
    }
    //copy data from GPU to CPU
    err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice); if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    err = cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);  if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    //timer start
    cudaEventRecord(start, 0);
    // Calculating inverse using Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        gaussjordan << <numBlocks, threadsPerBlock >> > (d_A, dI, n, i);
    }
    dev << <numBlocks, threadsPerBlock >> > (d_A, dI, n);

    err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost); if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    err = cudaMemcpy(L, d_A, ddsize, cudaMemcpyDeviceToHost); if (err != cudaSuccess) { cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Cuda Time - inverse: " << time << "ms\n";
    savetofile(iL, "inverse.txt", n, n);
    cudaFree(d_A);
    cudaFree(dI);
    delete[]I;
    delete[]L;
    delete[]iL;
    system("Pause");
    return 0;
}