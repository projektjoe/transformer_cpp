#define TESTING
#include "main.cpp"
#include <iostream>
#include <cmath>  // For std::abs

// Declare functions

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

    naive_matmul(out, x, y, x_row, x_col, y_col);

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
void test_quantized_matmul() {
    constexpr int x_row = 4;
    constexpr int x_col = 4;
    constexpr int y_row = 4;
    constexpr int y_col = 4;

    float x[x_row * x_col] = {2, 1, 0, 3, 1, 4, 2, 1, 0, 3, 1, 4, 2, 1, 0, 3};
    float y[y_row * y_col] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    float out_regular[x_row * y_col] = {0};
    float out_quantized[x_row * y_col] = {0};

    // Perform regular matrix multiplication
    naive_matmul(out_regular, x, y, x_row, x_col, y_col);

    // Quantize inputs
    QuantizedTensor qx, qy;
    qx.q = new int8_t[x_row * x_col];
    qx.s = new float[(x_row * x_col + GS - 1) / GS];
    qy.q = new int8_t[y_row * y_col];
    qy.s = new float[(y_row * y_col + GS - 1) / GS];

    quantize(&qx, x, x_row * x_col);
    quantize(&qy, y, y_row * y_col);

    // Perform quantized matrix multiplication
    naive_matmul_quantized(out_quantized, &qx, &qy, x_row, x_col, y_col);

    // Compare results
    bool test_passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < x_row * y_col; i++) {
        float diff = std::abs(out_regular[i] - out_quantized[i]);
        max_diff = std::max(max_diff, diff);
        if (!float_equal(out_regular[i], out_quantized[i], 0.1f)) {
            std::cout << "Quantized matmul test failed at index " << i << ": "
                      << "expected " << out_regular[i] << ", got " << out_quantized[i] << std::endl;
            test_passed = false;
        }
    }

    if (test_passed) {
        std::cout << "Quantized matmul test passed. Maximum difference: " << max_diff << std::endl;
    } else {
        std::cout << "Quantized matmul test failed. Maximum difference: " << max_diff << std::endl;
    }

    // Clean up
    delete[] qx.q;
    delete[] qx.s;
    delete[] qy.q;
    delete[] qy.s;
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

void test_tokenizer(){
    Tokenizer tokenizer("../llama2_model_weights/tokenizer.bin", 32000);  // Adjust vocab size as needed

    std::string text = "Hello, world!";
    std::vector<int> tokens = tokenizer.encode(text);

    std::cout << "Encoded tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    std::cout << "Decoded text: ";
    int prev_token = 1;  // Assume BOS token
    for (int token : tokens) {
        std::string piece = tokenizer.decode(prev_token, token);
        Tokenizer::safePrintf(piece);
        prev_token = token;
    }
    std::cout << std::endl;
}
int main() {
    test_naive_matmul();
    test_softmax();
    test_tokenizer();
    test_quantized_matmul();
    return 0;
}