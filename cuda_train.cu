#include <stdlib.h>
#include <stdio.h>
#include "cuda_train.h"
#include "cuda_utils.h"

__global__ void train_forward_layer(float *X, float *W, float *bias, float *y,
                                    int input_size, int output_size,
                                    ActivationFunction activation) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < output_size) {
        float sum = bias[col];
        for (int k = 0; k < input_size; ++k) {
            sum += X[k] * W[k * output_size + col];
        }
        y[col] = apply_activation(sum, activation);
    }
}

// Softmax needs the whole output vector, so one thread normalizes the vector after
// the final dense layer has written its raw logits.
__global__ void train_softmax_vector(float *vector, int n) {
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

// For softmax + cross-entropy, the output delta simplifies to output - target.
// Other activations still need their elementwise derivative.
__global__ void output_delta_kernel(float *output, float *delta, unsigned char label,
                                    int output_size, ActivationFunction activation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < output_size) {
        float target = (i == (int)label) ? 1.0f : 0.0f;
        float error = output[i] - target;
        delta[i] = (activation == ACTIVATION_SOFTMAX)
            ? error
            : error * activation_derivative_from_output(output[i], activation);
    }
}

// Backpropagate error from layer L+1 into layer L using the next layer's weights.
__global__ void hidden_delta_kernel(float *next_delta, float *next_W, float *layer_output,
                                    float *delta, int layer_output_size,
                                    int next_output_size, ActivationFunction activation) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < layer_output_size) {
        float error = 0.0f;
        for (int k = 0; k < next_output_size; ++k) {
            error += next_delta[k] * next_W[j * next_output_size + k];
        }
        delta[j] = error * activation_derivative_from_output(layer_output[j], activation);
    }
}

// One CUDA thread updates one weight. W is stored row-major:
// W[input_index * output_size + output_index].
__global__ void update_weights_kernel(float *input, float *delta, float *W,
                                      int input_size, int output_size,
                                      float learning_rate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int weight_count = input_size * output_size;

    if (index < weight_count) {
        int row = index / output_size;
        int col = index % output_size;
        W[index] -= learning_rate * input[row] * delta[col];
    }
}

__global__ void update_bias_kernel(float *delta, float *bias, int output_size,
                                   float learning_rate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < output_size) {
        bias[col] -= learning_rate * delta[col];
    }
}

typedef struct {
    // activations[0] is the input image. activations[i + 1] is layer i output.
    float **activations;
    // deltas[i] is the backprop error signal for layer i output neurons.
    float **deltas;
    float *host_output;
} GpuTrainingBuffers;

static void free_gpu_training_buffers(GpuTrainingBuffers *buffers, NeuralNetwork *network) {
    if (buffers == NULL) {
        return;
    }

    if (buffers->activations != NULL) {
        for (int i = 0; i <= network->number_of_layers; ++i) {
            cudaFree(buffers->activations[i]);
        }
        free(buffers->activations);
    }

    if (buffers->deltas != NULL) {
        for (int i = 0; i < network->number_of_layers; ++i) {
            cudaFree(buffers->deltas[i]);
        }
        free(buffers->deltas);
    }

    free(buffers->host_output);
}

static int allocate_gpu_training_buffers(NeuralNetwork *network, GpuTrainingBuffers *buffers) {
    buffers->activations = (float **)calloc(network->number_of_layers + 1, sizeof(float *));
    buffers->deltas = (float **)calloc(network->number_of_layers, sizeof(float *));
    buffers->host_output = (float *)malloc(network->output_size * sizeof(float));

    if (buffers->activations == NULL || buffers->deltas == NULL || buffers->host_output == NULL) {
        free_gpu_training_buffers(buffers, network);
        return 1;
    }

    CHECK_CUDA(cudaMalloc(&buffers->activations[0], network->input_size * sizeof(float)));

    for (int i = 0; i < network->number_of_layers; ++i) {
        int output_size = network->layers[i].output_size;
        CHECK_CUDA(cudaMalloc(&buffers->activations[i + 1], output_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&buffers->deltas[i], output_size * sizeof(float)));
    }

    return 0;
}

static int forward_gpu(NeuralNetwork *network, GpuTrainingBuffers *buffers) {
    for (int layer_index = 0; layer_index < network->number_of_layers; ++layer_index) {
        NeuralNetworkLayer *layer = &network->layers[layer_index];
        dim3 blockSize(256);
        dim3 gridSize((layer->output_size + blockSize.x - 1) / blockSize.x);

        train_forward_layer<<<gridSize, blockSize>>>(buffers->activations[layer_index],
                                                     network->d_W[layer_index],
                                                     network->d_bias[layer_index],
                                                     buffers->activations[layer_index + 1],
                                                     layer->input_size,
                                                     layer->output_size,
                                                     layer->activation);
        CHECK_CUDA(cudaGetLastError());
        if (layer->activation == ACTIVATION_SOFTMAX) {
            // Softmax is applied as a second pass over the final layer logits.
            train_softmax_vector<<<1, 1>>>(buffers->activations[layer_index + 1],
                                          layer->output_size);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

static int backward_gpu(NeuralNetwork *network, GpuTrainingBuffers *buffers,
                        unsigned char label) {
    int last_layer_index = network->number_of_layers - 1;
    NeuralNetworkLayer *last_layer = &network->layers[last_layer_index];
    dim3 blockSize(256);
    dim3 outputGrid((last_layer->output_size + blockSize.x - 1) / blockSize.x);

    // Start at the output layer, then walk backward through hidden layers.
    output_delta_kernel<<<outputGrid, blockSize>>>(buffers->activations[network->number_of_layers],
                                                  buffers->deltas[last_layer_index],
                                                  label,
                                                  last_layer->output_size,
                                                  last_layer->activation);
    CHECK_CUDA(cudaGetLastError());

    for (int layer_index = network->number_of_layers - 2; layer_index >= 0; --layer_index) {
        NeuralNetworkLayer *layer = &network->layers[layer_index];
        NeuralNetworkLayer *next_layer = &network->layers[layer_index + 1];
        dim3 gridSize((layer->output_size + blockSize.x - 1) / blockSize.x);

        hidden_delta_kernel<<<gridSize, blockSize>>>(buffers->deltas[layer_index + 1],
                                                    network->d_W[layer_index + 1],
                                                    buffers->activations[layer_index + 1],
                                                    buffers->deltas[layer_index],
                                                    layer->output_size,
                                                    next_layer->output_size,
                                                    layer->activation);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

static int update_weights_gpu(NeuralNetwork *network, GpuTrainingBuffers *buffers,
                              float learning_rate) {
    dim3 blockSize(256);

    for (int layer_index = 0; layer_index < network->number_of_layers; ++layer_index) {
        NeuralNetworkLayer *layer = &network->layers[layer_index];
        int weight_count = layer->input_size * layer->output_size;
        dim3 weightGrid((weight_count + blockSize.x - 1) / blockSize.x);
        dim3 biasGrid((layer->output_size + blockSize.x - 1) / blockSize.x);

        update_weights_kernel<<<weightGrid, blockSize>>>(buffers->activations[layer_index],
                                                        buffers->deltas[layer_index],
                                                        network->d_W[layer_index],
                                                        layer->input_size,
                                                        layer->output_size,
                                                        learning_rate);
        CHECK_CUDA(cudaGetLastError());

        update_bias_kernel<<<biasGrid, blockSize>>>(buffers->deltas[layer_index],
                                                   network->d_bias[layer_index],
                                                   layer->output_size,
                                                   learning_rate);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

static int predict_from_host_output(float *output, int output_size) {
    int prediction = 0;
    for (int i = 1; i < output_size; ++i) {
        if (output[i] > output[prediction]) {
            prediction = i;
        }
    }
    return prediction;
}

int train_mnist(NeuralNetwork *network, TrainConfig config) {
    if (network == NULL || config.image_path == NULL || config.label_path == NULL ||
        config.epochs <= 0 || config.samples_per_epoch <= 0 || config.learning_rate <= 0.0f) {
        printf("Invalid training configuration\n");
        return 1;
    }

    float *host_input = (float *)malloc(network->input_size * sizeof(float));
    if (host_input == NULL) {
        printf("Failed to allocate training input buffer\n");
        return 1;
    }

    GpuTrainingBuffers buffers;
    buffers.activations = NULL;
    buffers.deltas = NULL;
    buffers.host_output = NULL;

    if (allocate_gpu_training_buffers(network, &buffers) != 0) {
        printf("Failed to allocate GPU training buffers\n");
        free(host_input);
        return 1;
    }

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        int correct = 0;

        for (int sample = 0; sample < config.samples_per_epoch; ++sample) {
            unsigned char label = 0;
            // Dataset I/O stays on the host; the neural math below runs on the GPU.
            if (load_mnist_image(config.image_path, sample, host_input, network->input_size) != 0 ||
                load_mnist_label(config.label_path, sample, &label) != 0) {
                free_gpu_training_buffers(&buffers, network);
                free(host_input);
                return 1;
            }

            CHECK_CUDA(cudaMemcpy(buffers.activations[0], host_input,
                                  network->input_size * sizeof(float),
                                  cudaMemcpyHostToDevice));

            if (forward_gpu(network, &buffers) != 0) {
                free_gpu_training_buffers(&buffers, network);
                free(host_input);
                return 1;
            }

            CHECK_CUDA(cudaMemcpy(buffers.host_output,
                                  buffers.activations[network->number_of_layers],
                                  network->output_size * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            if (predict_from_host_output(buffers.host_output, network->output_size) == (int)label) {
                ++correct;
            }

            if (backward_gpu(network, &buffers, label) != 0 ||
                update_weights_gpu(network, &buffers, config.learning_rate) != 0) {
                free_gpu_training_buffers(&buffers, network);
                free(host_input);
                return 1;
            }
        }

        printf("Epoch %d/%d accuracy on training slice: %.2f%%\n",
               epoch + 1, config.epochs,
               ((float)correct / (float)config.samples_per_epoch) * 100.0f);
    }

    free_gpu_training_buffers(&buffers, network);
    free(host_input);
    return 0;
}
