#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#include "cuda_nn.h"

typedef struct {
    int training_samples;
    int training_epochs;
    int test_samples;
    float learning_rate;
    ActivationFunction activation;
    ActivationFunction final_activation;
} AppConfig;

AppConfig default_app_config(void);
int load_app_config(const char *path, AppConfig *config);
const char *activation_name(ActivationFunction activation);

#endif
