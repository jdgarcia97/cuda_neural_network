#include <stdio.h>
#include <string.h>
#include "nn_config.h"

static ActivationFunction parse_activation(const char *value) {
    if (strcmp(value, "relu") == 0) {
        return ACTIVATION_RELU;
    }
    if (strcmp(value, "logistic") == 0 || strcmp(value, "sigmoid") == 0) {
        return ACTIVATION_LOGISTIC;
    }
    if (strcmp(value, "softmax") == 0) {
        return ACTIVATION_SOFTMAX;
    }
    if (strcmp(value, "linear") == 0) {
        return ACTIVATION_LINEAR;
    }

    printf("Unknown activation '%s', using linear\n", value);
    return ACTIVATION_LINEAR;
}

const char *activation_name(ActivationFunction activation) {
    switch (activation) {
        case ACTIVATION_RELU:
            return "relu";
        case ACTIVATION_LOGISTIC:
            return "logistic";
        case ACTIVATION_SOFTMAX:
            return "softmax";
        case ACTIVATION_LINEAR:
        default:
            return "linear";
    }
}

AppConfig default_app_config(void) {
    AppConfig config;
    config.training_samples = 30000;
    config.training_epochs = 20;
    config.test_samples = 10000;
    config.learning_rate = 0.01f;
    config.activation = ACTIVATION_RELU;
    config.final_activation = ACTIVATION_SOFTMAX;
    return config;
}

int load_app_config(const char *path, AppConfig *config) {
    char key[64];
    char value[64];
    char line[256];

    if (path == NULL || config == NULL) {
        return 1;
    }

    FILE *file = fopen(path, "r");
    if (file == NULL) {
        printf("Config file not found, using defaults: %s\n", path);
        return 0;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }

        if (sscanf(line, " %63[^=]= %63s", key, value) != 2) {
            continue;
        }

        if (strcmp(key, "training_samples") == 0) {
            sscanf(value, "%d", &config->training_samples);
        } else if (strcmp(key, "training_epochs") == 0) {
            sscanf(value, "%d", &config->training_epochs);
        } else if (strcmp(key, "test_samples") == 0) {
            sscanf(value, "%d", &config->test_samples);
        } else if (strcmp(key, "learning_rate") == 0) {
            sscanf(value, "%f", &config->learning_rate);
        } else if (strcmp(key, "activation") == 0) {
            config->activation = parse_activation(value);
        } else if (strcmp(key, "final_activation") == 0) {
            config->final_activation = parse_activation(value);
        }
    }

    fclose(file);
    return 0;
}
