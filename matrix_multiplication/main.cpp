#include <iostream>

using namespace std;

int main ()
{
    int R1 = 2;
    int C1 = 4;
    int R2 = C1;
    int C2 = 3;
    
    int A[R1][C1] = {{1,2,1,4}, {3,1,1,5}};
    int B[R2][C2] = {{1,2,3}, {4,5,6}, {7,8,9}, {0,1,2}};
    
    int C[R1][C2];
    
    for (int i = 0; i < R1; i++)
    {
        for(int j = 0; j < C2; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < R2; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    for (int i = 0; i < R1; i++)
    {
        for (int j = 0; j < C2; j++)
        {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }
    
}