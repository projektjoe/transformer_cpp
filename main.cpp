#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

void naive_matmul(float *out, float *x, float *y, int x_row, int x_col, int y_row, int y_col)
{
    for (int i = 0; i < x_row; i++)
    {
        for (int j = 0; j < y_col; j++)
        {
            for (int k = 0; k < x_col; k++)
            {
                out[(i * y_col) + j] += x[(i * x_col) + k] * y[(y_col * k) + j];
            }
        }
    }
}


void softmax(float *out, float *input, int row, int col)
{

    for (int i = 0; i < row; i++)
    {
        // for every row, sum the exp of items
        float exp_sum = 0;
        auto* exp_mem = new float[col]();

        for (int j = 0; j < col; j++)
        {
            exp_mem[j] = expf(input[(i*col)+j]);
            exp_sum += expf(input[(i*col)+j]);
        }
        for (int j = 0; j < col; j++)
        {
            out[(i*col)+j] = exp_mem[j]/exp_sum;
        }
        delete[] exp_mem;
    }

}

void self_attn(float *out, float *K, float *Q, float *V, int T, int C, int head_size)
{
}

void attention_forward(float* out, float* input, int B, int T, int C, int head_size){

}
void attention_layer(float *out, float *input, int T, int C, int head_size){
    // generate K, Q, V from input
    float *K;
    float *Q;
    float *V; // (T,C)
    float *Wk;
    float *Wq;
    float *Wv;                              // (T,)
    naive_matmul(K, Wk, input, 3, 4, 4, 5); // (C, C) x (C, T), where inner dim should be divided into heads.
    naive_matmul(Q, Wq, input, 3, 4, 5, 6);
    naive_matmul(V, Wv, input, 3, 4, 5, 6);
    // self_attn(out, K, Q, V);
}

#ifndef TESTING
int main()
{

}
#endif
