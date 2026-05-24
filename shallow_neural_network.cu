#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_nn.h"
#include "cuda_test.h"
#include "cuda_train.h"
#include "nn_config.h"

// MNIST images are 28 x 28 grayscale pixels, and digit labels are 0-9.
#define INPUT_SIZE 784
#define HIDDEN_SIZE_1 128
#define HIDDEN_SIZE_2 64
#define OUTPUT_SIZE 10
#define NUMBER_OF_LAYERS 3
#define MNIST_IMAGE_INDEX 0
#define CONFIG_PATH "mnist.conf"
#define MNIST_TRAIN_IMAGE_PATH "data/mnist/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABEL_PATH "data/mnist/train-labels-idx1-ubyte"
#define MNIST_IMAGE_PATH "data/mnist/t10k-images-idx3-ubyte"
#define MNIST_LABEL_PATH "data/mnist/t10k-labels-idx1-ubyte"

int main() {
    float *X, *Y;
    int input_size = INPUT_SIZE * sizeof(float);
    int output_size = OUTPUT_SIZE * sizeof(float);

    srand(time(NULL));

    AppConfig app_config = default_app_config();
    if (load_app_config(CONFIG_PATH, &app_config) != 0) {
        return 1;
    }

    printf("Config: training_samples=%d training_epochs=%d test_samples=%d learning_rate=%.5f activation=%s final_activation=%s\n",
           app_config.training_samples,
           app_config.training_epochs,
           app_config.test_samples,
           app_config.learning_rate,
           activation_name(app_config.activation),
           activation_name(app_config.final_activation));

    // Allocate host memory
    X = (float *)malloc(input_size);
    Y = (float *)malloc(output_size);

    if (X == NULL || Y == NULL) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    if (load_mnist_image(MNIST_IMAGE_PATH, MNIST_IMAGE_INDEX, X, INPUT_SIZE) != 0) {
        free(X);
        free(Y);
        return 1;
    }

    unsigned char label = 0;
    if (load_mnist_label(MNIST_LABEL_PATH, MNIST_IMAGE_INDEX, &label) != 0) {
        free(X);
        free(Y);
        return 1;
    }

    // This describes a dense network: 784 -> 128 -> 64 -> 10.
    int layer_sizes[] = {INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE};
    ActivationFunction activations[] = {
        app_config.activation,
        app_config.activation,
        app_config.final_activation
    };

    NeuralNetwork *network = create_neural_network(NUMBER_OF_LAYERS, layer_sizes, activations);
    if (network == NULL) {
        free(X);
        free(Y);
        return 1;
    }

    TrainConfig train_config = {
        MNIST_TRAIN_IMAGE_PATH,
        MNIST_TRAIN_LABEL_PATH,
        app_config.training_epochs,
        app_config.training_samples,
        app_config.learning_rate
    };

    if (train_mnist(network, train_config) != 0) {
        free_neural_network(network);
        free(X);
        free(Y);
        return 1;
    }

    TestConfig test_config = {
        MNIST_IMAGE_PATH,
        MNIST_LABEL_PATH,
        app_config.test_samples
    };
    TestResult test_result;

    if (test_mnist(network, test_config, &test_result) != 0) {
        free_neural_network(network);
        free(X);
        free(Y);
        return 1;
    }

    printf("Test accuracy: %.2f%% (%d/%d)\n",
           test_result.accuracy, test_result.correct, test_result.total);
    printf("Test mean squared error: %.4f\n", test_result.mean_squared_error);

    if (neural_network_predict(network, X, Y) != 0) {
        free_neural_network(network);
        free(X);
        free(Y);
        return 1;
    }

    // Print the result (for verification)
    printf("MNIST label: %d\n", label);
    printf("Network digit scores:\n");
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        printf("%.4f ", Y[i]);
    }
    printf("\nPredicted digit: %d\n", predict_digit(Y, OUTPUT_SIZE));

    free_neural_network(network);
    free(X);
    free(Y);

    return 0;
}
