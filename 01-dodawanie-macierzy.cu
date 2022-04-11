%%cu
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


void initialInt(float *ip, float size)
{
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xff) / 66.6;
	}
}

void printMatrix(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A, *ib = B, *ic = C;
	printf("\nMatrix:(%d, %d)\n", nx, ny);
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%f + %f = %f     ", ia[ix], ib[ix], ic[ix]);
		}
		ia += nx;
		ib += nx;
		ic += nx;
		printf("\n");
	}
	printf("\n");
}
 
void printResult(float *C, const int nx, const int ny)
{
	float *ic = C;
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%f     ", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
}
 

__global__ void sumMatrixOnDevice(float *MatA, float *MatB, float *MatC, const int nx, const int ny)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;
	int iy = threadIdx.y + blockDim.y*blockIdx.y;
	unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny)
	{
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}

int main(int argc, char **argv)
{

	int nx = 3;
	int ny = 3;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	float *h_A, *h_B, *h_C;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);
 

	initialInt(h_A, nxy);
	initialInt(h_B, nxy);
	

	// mallox device memory
	float *d_MatA, *d_MatB, *d_MatC;
	cudaMalloc((void **)&d_MatA, nBytes);
	cudaMalloc((void **)&d_MatB, nBytes);
	cudaMalloc((void **)&d_MatC, nBytes);

	 
	// transfer data from host
	cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(nx, ny);
	sumMatrixOnDevice <<<1,  threadsPerBlock>>> (d_MatA, d_MatB, d_MatC, nx, ny);


	// transfer data from device
	cudaMemcpy(h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);

	printMatrix(h_A, h_B, h_C, nx, ny);

	printResult(h_C, nx, ny);


	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);


	cudaDeviceReset();

	return 0;
}