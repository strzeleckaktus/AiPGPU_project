// Matrix_Multiplication_Sequential2.cpp : Ten plik zawiera funkcję „main”. W nim rozpoczyna się i kończy wykonywanie programu.
//

#include <iostream>
using namespace std;

int main()
{
    const int matrix_width = 3;
    const int matrix_height = 3;
    float matrix[matrix_width][matrix_height];

    for (int i = 0; i < matrix_width; i++)
    {
        cout << "[ ";
        for (int j = 0; j < matrix_height; j++)
        {
            matrix[i][j] = rand() % 123;
            cout << matrix[i][j] << " ";
        }
        cout << "]" << endl;
    }

    cout << endl;
    const int multiplier = 2;

    for (int i = 0; i < matrix_width; i++)
    {
        cout << "[ ";
        for (int j = 0; j < matrix_height; j++)
        {
            matrix[i][j] *= multiplier;
            cout << matrix[i][j] << " ";
        }
        cout << "]" << endl;
    }
}
