#include <stdlib.h>
#include <stdio.h>
#include "cuda_test.h"

int predict_digit(float *scores, int output_size) {
    int prediction = 0;
    for (int i = 1; i < output_size; ++i) {
        if (scores[i] > scores[prediction]) {
            prediction = i;
        }
    }
    return prediction;
}

static float mse_for_label(float *scores, int output_size, unsigned char label) {
    float sum = 0.0f;

    for (int i = 0; i < output_size; ++i) {
        float target = (i == (int)label) ? 1.0f : 0.0f;
        float error = scores[i] - target;
        sum += error * error;
    }

    return sum / (float)output_size;
}

int test_mnist(NeuralNetwork *network, TestConfig config, TestResult *result) {
    if (network == NULL || config.image_path == NULL || config.label_path == NULL ||
        config.samples <= 0 || result == NULL) {
        printf("Invalid test configuration\n");
        return 1;
    }

    float *X = (float *)malloc(network->input_size * sizeof(float));
    float *Y = (float *)malloc(network->output_size * sizeof(float));
    if (X == NULL || Y == NULL) {
        printf("Failed to allocate test buffers\n");
        free(X);
        free(Y);
        return 1;
    }

    int correct = 0;
    float total_mse = 0.0f;

    for (int sample = 0; sample < config.samples; ++sample) {
        unsigned char label = 0;
        // Testing is read-only: load image, run inference, compare argmax to label.
        if (load_mnist_image(config.image_path, sample, X, network->input_size) != 0 ||
            load_mnist_label(config.label_path, sample, &label) != 0 ||
            neural_network_predict(network, X, Y) != 0) {
            free(X);
            free(Y);
            return 1;
        }

        int prediction = predict_digit(Y, network->output_size);
        if (prediction == (int)label) {
            ++correct;
        }

        total_mse += mse_for_label(Y, network->output_size, label);
    }

    result->correct = correct;
    result->total = config.samples;
    result->accuracy = ((float)correct / (float)config.samples) * 100.0f;
    result->mean_squared_error = total_mse / (float)config.samples;

    free(X);
    free(Y);
    return 0;
}
