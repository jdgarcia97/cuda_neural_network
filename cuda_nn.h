#ifndef CUDA_NN_H
#define CUDA_NN_H

#include <math.h>

typedef enum {
    ACTIVATION_LINEAR = 0,
    ACTIVATION_RELU = 1,
    ACTIVATION_LOGISTIC = 2,
    ACTIVATION_SOFTMAX = 3
} ActivationFunction;

typedef struct {
    int input_size;
    int output_size;
    float *W;
    float *bias;
    ActivationFunction activation;
} NeuralNetworkLayer;

typedef struct {
    int number_of_layers;
    int input_size;
    int output_size;
    NeuralNetworkLayer *layers;
    float **d_W;
    float **d_bias;
    float *d_hidden_a;
    float *d_hidden_b;
} NeuralNetwork;

// Small activation helpers are inline so host code and device kernels can share
// the same activation behavior without separate CPU/GPU implementations.
__host__ __device__ static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__host__ __device__ static inline float logistic(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ static inline float apply_activation(float x, ActivationFunction activation) {
    switch (activation) {
        case ACTIVATION_RELU:
            return relu(x);
        case ACTIVATION_LOGISTIC:
            return logistic(x);
        case ACTIVATION_SOFTMAX:
        case ACTIVATION_LINEAR:
        default:
            return x;
    }
}

__host__ __device__ static inline float activation_derivative_from_output(float y, ActivationFunction activation) {
    switch (activation) {
        case ACTIVATION_RELU:
            return y > 0.0f ? 1.0f : 0.0f;
        case ACTIVATION_LOGISTIC:
            return y * (1.0f - y);
        case ACTIVATION_SOFTMAX:
        case ACTIVATION_LINEAR:
        default:
            return 1.0f;
    }
}

// Normalize the input vector to have values between 0 and 1
__host__ __device__ static inline void normalize_vector(float *vector, int n) {
    if (vector == 0 || n <= 0) {
        return;
    }

    float max_val = vector[0];
    float min_val = vector[0];
    for (int i = 1; i < n; i++) {
        if (vector[i] > max_val) {
            max_val = vector[i];
        }
        if (vector[i] < min_val) {
            min_val = vector[i];
        }           
    }
    float range = max_val - min_val;
    if (range > 0.0f) {
        for (int i = 0; i < n; i++) {
            vector[i] = (vector[i] - min_val) / range;
        }
    }
}


void generate_random_vector(float *vector, int n);
void generate_random_matrix(float *matrix, int rows, int cols);
NeuralNetworkLayer *generate_neural_network_layers(int number_of_layers, const int *layer_sizes,
                                                   const ActivationFunction *activations);
void free_neural_network_layers(NeuralNetworkLayer *layers, int number_of_layers);
NeuralNetwork *create_neural_network(int number_of_layers, const int *layer_sizes,
                                     const ActivationFunction *activations);
int neural_network_sync_to_device(NeuralNetwork *network);
int neural_network_predict(NeuralNetwork *network, float *X, float *Y);
void free_neural_network(NeuralNetwork *network);
int load_mnist_image(const char *image_path, int image_index, float *image, int image_size);
int load_mnist_label(const char *label_path, int label_index, unsigned char *label);

#endif
