#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
#include <cstdlib>
#include<cuda.h> 
#include<stdio.h> 
#include<math.h> 
#include<conio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void Kernel(float* A_, float* B_, int size)
{
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    __shared__ float temp[16][16];

    //Copying the data to the shared memory 
    temp[idy][idx] = A_[(idy * (size + 1)) + idx];

    for (int i = 1; i < size; i++)
    {
        if ((idy + i) < size) // NO Thread divergence here 
        {
            float var1 = temp[i - 1][i - 1] / temp[i + idy][i - 1];                         //obliczamy stosunek między konkretnymi wierszami
            temp[i + idy][idx] = temp[i - 1][idx] - ((var1) * (temp[i + idy][idx]));        //Mnożymy konkretne elementy wierszy przez obliczony wcześniej stosunek
        }
        __syncthreads(); //Synchronizing all threads before Next iterat ion 
    }

    B_[idy * (size + 1) + idx] = temp[idy][idx];
}

// copying the value from file to array 
void copyvalue(int newchar, int* i, FILE* data, float* temp_h)
{
    float sum, sumfrac;
    double count;
    int ch;
    int ptPresent = 0;
    float sign;

    sum = sumfrac = 0.0;
    count = 1.0;

    if (newchar == '-')
    {
        sign = -1.0;
        fgetc(data);
    }
    else
    {
        sign = 1.0;
    }

    while (1)
    {
        ch = fgetc(data);

        if (ch == '\n' || ch == ' ')
        {
            ungetc(ch, data);
            break;
        }
        else if (ch == '.')
        {
            ptPresent = 1;
            break;
        }
        else
        {
            sum = sum * 10 + ch - 48;
        }
    }

    if (ptPresent)
    {
        while (1)
        {
            ch = fgetc(data);
            if (ch == ' ' || ch == '\n')
            {
                ungetc(ch, data);
                break;
            }
            else
            {
                sumfrac = sumfrac + ((float)(ch - 48)) / pow(10.0, count);
                count++;
            }
        }
    }

    temp_h[*i] = sign * (sum + sumfrac);

    printf("[%f]", temp_h[*i]);
    (*i)++;
}

void DeviceFunc(float* temp_h, int variablesNo, float* temp1_h)
{
    float* A_, * B_;

    //Memory allocation on the device 
    cudaMalloc(&A_, sizeof(float) * (variablesNo) * (variablesNo + 1));
    cudaMalloc(&B_, sizeof(float) * (variablesNo) * (variablesNo + 1));

    //Copying data to device from host 
    cudaMemcpy(A_, temp_h, sizeof(float) * variablesNo * (variablesNo + 1), cudaMemcpyHostToDevice);

    //Defining size of Thread Block 
    dim3 dimBlock(variablesNo + 1, variablesNo, 1);
    dim3 dimGrid(1, 1, 1);

    //Kernel call 
    Kernel << <dimGrid, dimBlock >> > (A_, B_, variablesNo);

    //Coping data to host from device 
    cudaMemcpy(temp1_h, B_, sizeof(float) * variablesNo * (variablesNo + 1), cudaMemcpyDeviceToHost);

    //Deallocating memory on the device 
    cudaFree(A_);
    cudaFree(B_);
}

void getvalue(float** temp_h, int* variablesNo) {
    FILE* data;
    int newchar, index;

    index = 0;

    data = fopen("data3.txt", "r");

    if (data == NULL) // if file does not exist 
    {
        perror("data3.txt");
        exit(1);
    }

    //First line contains number of variables 
    while ((newchar = fgetc(data)) != '\n')
    {
        *variablesNo = (*variablesNo) * 10 + (newchar - 48);
    }

    printf("\nNumber of variables = %d\n", *variablesNo);

    //Allocating memory for the array to store coefficients 
    *temp_h = (float*)malloc(sizeof(float) * (*variablesNo) * (*variablesNo + 1));

    while (1)
    {
        //Reading the remaining data
        newchar = fgetc(data);

        if (feof(data))
        {
            break;
        }

        if (newchar == ' ')
        {
            printf(" ");
            continue;
        }
        else if (newchar == '\n')
        {
            printf("\n\n");
            continue;
        }
        else if ((newchar >= 48 && newchar <= 57) || newchar == '-')
        {
            ungetc(newchar, data);
            copyvalue(newchar, &index, data, *temp_h);
        }
        else {
            printf("\nError:Unexpected symbol %c found", newchar);
            _getch();
            exit(1);
        }
    }
}



int main(int argc, char** argv)
{
    float* A = NULL;
    float* B = NULL;
    float* result, sum, rValue;
    int variablesNo, j;

    variablesNo = 0;

    // Reading the file to copy values 
    printf("\t\tShowing the data read from file\n\n");
    getvalue(&A, &variablesNo);

    //Allocating memory on host for B 
    B = (float*)malloc(sizeof(float) * variablesNo * (variablesNo + 1));

    //Calling device function to copy data to device 
    DeviceFunc(A, variablesNo, B);

    //Showing the data 
    printf("\n\n");

    for (int i = 0; i < variablesNo; i++)
    {
        for (int j = 0; j < variablesNo + 1; j++)
        {
            printf("%f\n", B[i * (variablesNo + 1) + j]);
        }
        printf("\n");
    }

    //Using Back substitution method 
    result = (float*)malloc(sizeof(float) * (variablesNo));
    for (int i = 0; i < variablesNo; i++)
    {
        result[i] = 1.0;
    }

    for (int i = variablesNo - 1; i >= 0; i--)
    {
        sum = 0.0;

        for (j = variablesNo - 1; j > i; j--)
        {
            sum = sum + result[j] * B[i * (variablesNo + 1) + j];  //podstawiamy konkretne wartości pod zmienne
        }
        rValue = B[i * (variablesNo + 1) + variablesNo] - sum;  //odejmujemy z prawej strony równania
        result[i] = rValue / B[i * (variablesNo + 1) + j]; //dzielimy prawą stronę równania przez to co jest przy zmiennej
    }

    //Displaying the result 
    printf("\n\t\tVALUES OF VARIABLES\n\n");
    for (int i = 0; i < variablesNo; i++)
    {
        printf("[X%d] = %+f\n", i, result[i]);
    }

    system("pause");
    return 0;
}
