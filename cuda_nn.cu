#include <stdlib.h>
#include <stdio.h>
#include "cuda_nn.h"
#include "cuda_utils.h"

__global__ void neural_network_layer(float *X, float *W, float *bias, float *y,
                                     int input_size, int output_size,
                                     ActivationFunction activation) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < output_size) {
        float sum = 0.0f;
        for (int k = 0; k < input_size; ++k) {
            sum += X[k] * W[k * output_size + col];
        }
        y[col] = apply_activation(sum + bias[col], activation);
    }
}

// Inference softmax mirrors the training softmax: dense layer writes logits,
// then this kernel turns those logits into class probabilities.
__global__ void softmax_vector(float *vector, int n) {
    float max_val = vector[0];
    for (int i = 1; i < n; ++i) {
        if (vector[i] > max_val) {
            max_val = vector[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        vector[i] = expf(vector[i] - max_val);
        sum += vector[i];
    }

    for (int i = 0; i < n; ++i) {
        vector[i] /= sum;
    }
}

static float random_float(void) {
    return (((float)rand() / (float)RAND_MAX) * 0.2f) - 0.1f;
}

static int read_big_endian_int(FILE *file, int *value) {
    unsigned char bytes[4];
    if (fread(bytes, sizeof(unsigned char), 4, file) != 4) {
        return 1;
    }

    *value = ((int)bytes[0] << 24) |
             ((int)bytes[1] << 16) |
             ((int)bytes[2] << 8) |
             (int)bytes[3];
    return 0;
}

void generate_random_vector(float *vector, int n) {
    for (int i = 0; i < n; ++i) {
        vector[i] = random_float();
    }
}

void generate_random_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = random_float();
    }
}

NeuralNetworkLayer *generate_neural_network_layers(int number_of_layers, const int *layer_sizes,
                                                   const ActivationFunction *activations) {
    NeuralNetworkLayer *layers = (NeuralNetworkLayer *)calloc(number_of_layers, sizeof(NeuralNetworkLayer));
    if (layers == NULL) {
        printf("Failed to allocate memory for neural network layers\n");
        return NULL;
    }

    for (int i = 0; i < number_of_layers; ++i) {
        // layer_sizes has one more entry than layers: {input, hidden..., output}.
        layers[i].input_size = layer_sizes[i];
        layers[i].output_size = layer_sizes[i + 1];
        layers[i].activation = activations[i];
        layers[i].W = (float *)malloc(layers[i].input_size * layers[i].output_size * sizeof(float));
        layers[i].bias = (float *)malloc(layers[i].output_size * sizeof(float));

        if (layers[i].W == NULL || layers[i].bias == NULL) {
            printf("Failed to allocate memory for layer %d\n", i);
            free_neural_network_layers(layers, number_of_layers);
            return NULL;
        }

        generate_random_matrix(layers[i].W, layers[i].input_size, layers[i].output_size);
        generate_random_vector(layers[i].bias, layers[i].output_size);
    }

    return layers;
}

void free_neural_network_layers(NeuralNetworkLayer *layers, int number_of_layers) {
    if (layers == NULL) {
        return;
    }

    for (int i = 0; i < number_of_layers; ++i) {
        free(layers[i].W);
        free(layers[i].bias);
    }

    free(layers);
}

static int max_intermediate_size(NeuralNetwork *network) {
    int max_size = network->output_size;
    for (int i = 0; i < network->number_of_layers - 1; ++i) {
        if (network->layers[i].output_size > max_size) {
            max_size = network->layers[i].output_size;
        }
    }
    return max_size;
}

static int copy_layers_to_device(NeuralNetwork *network) {
    for (int i = 0; i < network->number_of_layers; ++i) {
        int weight_size = network->layers[i].input_size * network->layers[i].output_size * sizeof(float);
        int bias_size = network->layers[i].output_size * sizeof(float);

        CHECK_CUDA(cudaMalloc(&network->d_W[i], weight_size));
        CHECK_CUDA(cudaMalloc(&network->d_bias[i], bias_size));
        CHECK_CUDA(cudaMemcpy(network->d_W[i], network->layers[i].W,
                              weight_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(network->d_bias[i], network->layers[i].bias,
                              bias_size, cudaMemcpyHostToDevice));
    }

    return 0;
}

int neural_network_sync_to_device(NeuralNetwork *network) {
    if (network == NULL) {
        return 1;
    }

    for (int i = 0; i < network->number_of_layers; ++i) {
        int weight_size = network->layers[i].input_size * network->layers[i].output_size * sizeof(float);
        int bias_size = network->layers[i].output_size * sizeof(float);

        CHECK_CUDA(cudaMemcpy(network->d_W[i], network->layers[i].W,
                              weight_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(network->d_bias[i], network->layers[i].bias,
                              bias_size, cudaMemcpyHostToDevice));
    }

    return 0;
}

static int compute_neural_network(NeuralNetwork *network, float *d_X, float *d_Y) {
    float *layer_input = d_X;

    for (int i = 0; i < network->number_of_layers; ++i) {
        NeuralNetworkLayer *layer = &network->layers[i];
        dim3 blockSize(256);
        dim3 gridSize((layer->output_size + blockSize.x - 1) / blockSize.x);

        float *layer_output = d_Y;
        if (i < network->number_of_layers - 1) {
            // Reuse two hidden buffers while ping-ponging through hidden layers.
            layer_output = (i % 2 == 0) ? network->d_hidden_a : network->d_hidden_b;
        }

        neural_network_layer<<<gridSize, blockSize>>>(layer_input, network->d_W[i],
                                                      network->d_bias[i], layer_output,
                                                      layer->input_size, layer->output_size,
                                                      layer->activation);
        CHECK_CUDA(cudaGetLastError());
        if (layer->activation == ACTIVATION_SOFTMAX) {
            softmax_vector<<<1, 1>>>(layer_output, layer->output_size);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        layer_input = layer_output;
    }

    return 0;
}

NeuralNetwork *create_neural_network(int number_of_layers, const int *layer_sizes,
                                     const ActivationFunction *activations) {
    if (number_of_layers <= 0 || layer_sizes == NULL || activations == NULL) {
        printf("Invalid neural network configuration\n");
        return NULL;
    }

    for (int i = 0; i <= number_of_layers; ++i) {
        if (layer_sizes[i] <= 0) {
            printf("Invalid layer size at index %d\n", i);
            return NULL;
        }
    }

    NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (network == NULL) {
        printf("Failed to allocate neural network\n");
        return NULL;
    }

    network->number_of_layers = number_of_layers;
    network->input_size = layer_sizes[0];
    network->output_size = layer_sizes[number_of_layers];
    network->layers = generate_neural_network_layers(number_of_layers, layer_sizes, activations);
    network->d_W = (float **)calloc(number_of_layers, sizeof(float *));
    network->d_bias = (float **)calloc(number_of_layers, sizeof(float *));
    network->d_hidden_a = NULL;
    network->d_hidden_b = NULL;

    if (network->layers == NULL || network->d_W == NULL || network->d_bias == NULL) {
        free_neural_network(network);
        return NULL;
    }

    int hidden_size = max_intermediate_size(network) * sizeof(float);
    cudaError_t err = cudaMalloc(&network->d_hidden_a, hidden_size);
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        free_neural_network(network);
        return NULL;
    }

    err = cudaMalloc(&network->d_hidden_b, hidden_size);
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        free_neural_network(network);
        return NULL;
    }

    if (copy_layers_to_device(network) != 0) {
        free_neural_network(network);
        return NULL;
    }

    return network;
}

int neural_network_predict(NeuralNetwork *network, float *X, float *Y) {
    if (network == NULL || X == NULL || Y == NULL) {
        return 1;
    }

    int input_size = network->input_size * sizeof(float);
    int output_size = network->output_size * sizeof(float);
    float *d_X = NULL;
    float *d_Y = NULL;

    CHECK_CUDA(cudaMalloc(&d_X, input_size));
    CHECK_CUDA(cudaMalloc(&d_Y, output_size));
    CHECK_CUDA(cudaMemcpy(d_X, X, input_size, cudaMemcpyHostToDevice));

    if (compute_neural_network(network, d_X, d_Y) != 0) {
        cudaFree(d_X);
        cudaFree(d_Y);
        return 1;
    }

    CHECK_CUDA(cudaMemcpy(Y, d_Y, output_size, cudaMemcpyDeviceToHost));

    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}

void free_neural_network(NeuralNetwork *network) {
    if (network == NULL) {
        return;
    }

    if (network->d_W != NULL) {
        for (int i = 0; i < network->number_of_layers; ++i) {
            cudaFree(network->d_W[i]);
        }
    }

    if (network->d_bias != NULL) {
        for (int i = 0; i < network->number_of_layers; ++i) {
            cudaFree(network->d_bias[i]);
        }
    }

    cudaFree(network->d_hidden_a);
    cudaFree(network->d_hidden_b);

    free(network->d_W);
    free(network->d_bias);
    free_neural_network_layers(network->layers, network->number_of_layers);
    free(network);
}

int load_mnist_image(const char *image_path, int image_index, float *image, int image_size) {
    FILE *file = fopen(image_path, "rb");
    if (file == NULL) {
        printf("Failed to open MNIST image file: %s\n", image_path);
        return 1;
    }

    // MNIST IDX files store header integers in big-endian byte order.
    int magic, number_of_images, rows, cols;
    if (read_big_endian_int(file, &magic) ||
        read_big_endian_int(file, &number_of_images) ||
        read_big_endian_int(file, &rows) ||
        read_big_endian_int(file, &cols)) {
        printf("Failed to read MNIST image header\n");
        fclose(file);
        return 1;
    }

    if (magic != 2051 || image_index < 0 || image_index >= number_of_images ||
        rows * cols != image_size) {
        printf("Invalid MNIST image file or image index\n");
        fclose(file);
        return 1;
    }

    long offset = 16L + (long)image_index * image_size;
    if (fseek(file, offset, SEEK_SET) != 0) {
        printf("Failed to seek to MNIST image\n");
        fclose(file);
        return 1;
    }

    for (int i = 0; i < image_size; ++i) {
        unsigned char pixel;
        if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
            printf("Failed to read MNIST image pixels\n");
            fclose(file);
            return 1;
        }
        image[i] = (float)pixel / 255.0f;
    }

    fclose(file);
    return 0;
}

int load_mnist_label(const char *label_path, int label_index, unsigned char *label) {
    FILE *file = fopen(label_path, "rb");
    if (file == NULL) {
        printf("Failed to open MNIST label file: %s\n", label_path);
        return 1;
    }

    int magic, number_of_labels;
    if (read_big_endian_int(file, &magic) ||
        read_big_endian_int(file, &number_of_labels)) {
        printf("Failed to read MNIST label header\n");
        fclose(file);
        return 1;
    }

    if (magic != 2049 || label_index < 0 || label_index >= number_of_labels) {
        printf("Invalid MNIST label file or label index\n");
        fclose(file);
        return 1;
    }

    long offset = 8L + (long)label_index;
    if (fseek(file, offset, SEEK_SET) != 0 ||
        fread(label, sizeof(unsigned char), 1, file) != 1) {
        printf("Failed to read MNIST label\n");
        fclose(file);
        return 1;
    }

    fclose(file);
    return 0;
}
