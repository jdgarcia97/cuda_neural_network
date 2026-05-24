NVCC ?= nvcc
NVCCFLAGS ?= -ccbin clang-14
LDLIBS ?= -lm

.PHONY: all clean mnist-data

all: shallow_neural_network 

shallow_neural_network: shallow_neural_network.cu cuda_nn.cu cuda_nn.h cuda_train.cu cuda_train.h cuda_test.cu cuda_test.h nn_config.cu nn_config.h cuda_utils.h
	$(NVCC) $(NVCCFLAGS) shallow_neural_network.cu cuda_nn.cu cuda_train.cu cuda_test.cu nn_config.cu -o shallow_neural_network $(LDLIBS)


clean:
	rm -f shallow_neural_network 

mnist-data:
	mkdir -p data/mnist
	curl -L -o data/mnist/train-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
	curl -L -o data/mnist/train-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
	curl -L -o data/mnist/t10k-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
	curl -L -o data/mnist/t10k-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
	gzip -dkf data/mnist/train-images-idx3-ubyte.gz
	gzip -dkf data/mnist/train-labels-idx1-ubyte.gz
	gzip -dkf data/mnist/t10k-images-idx3-ubyte.gz
	gzip -dkf data/mnist/t10k-labels-idx1-ubyte.gz
