#include <iostream>
using namespace std;


void transpose() {
    const int x = 3;
    const int y = 3;
    int input[x][y] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };

    int output[y][x];

    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            output[j][i] = input[i][j];
        }
    }
    for (int i = 0; i < y; i++)
    {
        for (int j = 0; j < x; j++)
        {
            cout << output[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    transpose();

    return 0;
}