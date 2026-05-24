#ifndef CUDA_TRAIN_H
#define CUDA_TRAIN_H

#include "cuda_nn.h"

typedef struct {
    const char *image_path;
    const char *label_path;
    int epochs;
    int samples_per_epoch;
    float learning_rate;
} TrainConfig;

int train_mnist(NeuralNetwork *network, TrainConfig config);

#endif
