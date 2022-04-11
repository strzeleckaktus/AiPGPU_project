%%cu


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define TILE_WIDTH 2

__global__ void add(float* a, float* b, float* c) {
	c[blockIdx.x * blockDim.x + threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x] + b[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void mul(float* Md, float* Nd, float* Pd, int Width) { 
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float w = 0.0;
	for (int k = 0; k < Width; k++) {
		w += Md[ty * Width + k] * Nd[k * Width + tx];
	}
	Pd[ty * Width + tx] = w;
}

__global__ void blockMul(float* Md, float* Nd, float* Pd, int Width) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockDim.y * Width * blockIdx.y + threadIdx.y * Width;
	int val = col+row;
	float w = 0.0;
	for (int k = 0; k < Width; k++) {
		w += Md[row + k] + Nd[col + k * Width];
	}
	Pd[val] = w;
}

__global__ void blockSharedMul(float* Md, float* Nd, float* Pd, int Width) {
	int col = threadIdx.x;
	int row = threadIdx.y;
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	float cValue = 0.0f;
	for (int m = 0; m < (Width / TILE_WIDTH); m++) {
		__shared__ float Ns[TILE_WIDTH][TILE_WIDTH];
		__shared__ float Ms[TILE_WIDTH][TILE_WIDTH];

		Ms[row][col] = Md[row*Width + TILE_WIDTH * Width * blockRow+ TILE_WIDTH *m + col];
		Ns[row][col] = Nd[row * Width + TILE_WIDTH * Width * m + TILE_WIDTH * blockCol + col];

		__syncthreads();

		for (int e = 0; e < TILE_WIDTH; ++e) {
			cValue += Ms[threadIdx.y][e] * Ns[e][threadIdx.x];
		}

		__syncthreads();
	}

	Pd[row * Width + TILE_WIDTH * Width * blockRow + TILE_WIDTH * blockCol + col] = cValue;
}

void MulOnHost(float* M, float* N, float* P, int Width) {
	for (int i = 0; i < Width; ++i) {
		for (int j = 0; j < Width; ++j) {
			float sum = 0;
			for (int k = 0; k < Width; ++k) {
				float a = M[i * Width + k];
				float b = N[k * Width + j];
				sum += a * b;
			}
			P[i * Width + j] = sum;
		}
	}
}
void printMatrix(float* M, int Width){
  std::cout <<"[";
  for (int i = 0; i < Width; i++) {
    for (int j = 0; j < Width; j++) {
      std::cout << M[i * Width + j] << "  ";
    }
    if (i != Width-1){
      std::cout <<"\n";
    }
  }
  std::cout <<"]\n";
}

int main(void) {
	srand(time(NULL));
	int Cols = 4;
  int Rows = 4;
  int TileWidth = TILE_WIDTH;
	float**M, **N, **P, **P_copy, *d_m, *d_n, *d_p;

	cudaMalloc(&d_m, Rows * Cols * sizeof(float));
	cudaMalloc(&d_n, Rows * Cols * sizeof(float));
	cudaMalloc(&d_p, Rows * Cols * sizeof(float));

	M = new float* [Rows];
	M[0] = new float[Cols * Rows];
	N = new float* [Rows];
	N[0] = new float[Cols * Rows];
	P = new float* [Rows];
	P[0] = new float[Cols * Rows];
	P_copy = new float* [Rows];
	P_copy[0] = new float[Cols * Rows];

	for (int i = 1; i < Rows; i++) {
		M[i] = M[0] + i * Cols;
		N[i] = N[0] + i * Cols;
		P[i] = P[0] + i * Cols;
		P_copy[i] = P_copy[0] + i * Cols;
	}

	for (int i = 0; i < Rows; i++) {
		for (int j = 0; j < Cols; j++) {
			M[i][j] = rand() % 5 + 1;
			N[i][j] = rand() % 5 + 1;
		}
	}

  std::cout << "Macierz A \n";
  printMatrix(M[0], Cols);
  std::cout << "\nMacierz B \n";
  printMatrix(N[0], Cols);

	
	cudaMemcpy(d_m, M[0], Rows * Cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, N[0], Rows * Cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, P[0], Rows * Cols * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(Cols / TileWidth, Rows / TileWidth, 1);
	dim3 block(TileWidth, TileWidth, 1);

	blockSharedMul <<<grid, block >>> (d_m, d_n, d_p, Cols);
	cudaMemcpy(P[0], d_p, Rows * Cols * sizeof(float), cudaMemcpyDeviceToHost);

	MulOnHost(M[0], N[0], P_copy[0], Cols);

  std::cout << "\nWYNIKI: \n\nMnożenie na host\n";
  printMatrix(P_copy[0], Cols);

  std::cout << "\n\nMnożenie GPU\n";
  printMatrix(P[0], Cols);

	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_p);

	delete[] N[0];
	delete[] N;
	delete[] P[0];
	delete[] P;
	delete[] M[0];
	delete[] M;
	delete[] P_copy[0];
	delete[] P_copy;

	return 0;
}