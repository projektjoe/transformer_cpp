#define TESTING
#include "main.cpp"
#include <iostream>
#include <cmath>  // For std::abs

// Declare functions
void naive_matmul(float *out, float *x, float *y, int x_row, int x_col, int y_row, int y_col);
void softmax(float *out, float *input, int row, int col);

// Helper function to compare floating-point numbers
bool float_equal(float a, float b, float epsilon = 1e-5) {
    return std::fabs(a - b) < epsilon;
}

void test_naive_matmul() {
    constexpr int x_row = 3;
    constexpr int x_col = 2;

    constexpr int y_row = 2;
    constexpr int y_col = 3;

    float x[x_row * x_col] = {2, 1, 0, 3, 1, 4};
    float y[y_row * y_col] = {1, 2, 3, 4, 5, 6};
    float out[x_row * y_col] = {0};
    float expected[x_row * y_col] = {6, 9, 12, 12, 17, 22, 9, 14, 19};

    naive_matmul(out, x, y, x_row, x_col, y_row, y_col);

    bool test_passed = true;
    for (int i = 0; i < x_row * y_col; i++) {
        if (!float_equal(out[i], expected[i])) {
            std::cout << "naive_matmul test failed at index " << i << ": "
                      << "expected " << expected[i] << ", got " << out[i] << std::endl;
            test_passed = false;
        }
    }
    if (test_passed) {
        std::cout << "naive_matmul test passed." << std::endl;
    }
}

void test_softmax() {
    constexpr int row = 2;
    constexpr int col = 3;

    float input[row * col] = {1, 2, 3, 1, 2, 3};
    float out[row * col] = {0};
    float expected[row * col] = {
            0.09003057, 0.24472847, 0.66524096,
            0.09003057, 0.24472847, 0.66524096};

    softmax(out, input, row, col);

    bool test_passed = true;
    for (int i = 0; i < row * col; i++) {
        if (!float_equal(out[i], expected[i])) {
            std::cout << "softmax test failed at index " << i << ": "
                      << "expected " << expected[i] << ", got " << out[i] << std::endl;
            test_passed = false;
        }
    }
    if (test_passed) {
        std::cout << "softmax test passed." << std::endl;
    }
}

int main() {
    test_naive_matmul();
    test_softmax();
    return 0;
}