#ifndef CUDA_TEST_H
#define CUDA_TEST_H

#include "cuda_nn.h"

typedef struct {
    const char *image_path;
    const char *label_path;
    int samples;
} TestConfig;

typedef struct {
    int correct;
    int total;
    float accuracy;
    float mean_squared_error;
} TestResult;

int predict_digit(float *scores, int output_size);
int test_mnist(NeuralNetwork *network, TestConfig config, TestResult *result);

#endif
