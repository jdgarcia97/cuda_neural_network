#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Multiply an N x N matrix. 
#define MATRIX_SIZE 100

// Lets create a cuda function for 
// a neural network. we multiply X*W + bias = y. 
// The input X is a 1 x N vector the weight W is a N x N matrix and the bias is a 1 x N vector. 
// The output y is a 1 x N vector.

__global__ void neural_network_layer(int *X, int *W, int *bias, int *y, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += X[k] * W[k * n + col];
        }
        y[col] = sum + bias[col];
    }
}

__global__ void matrix_mult(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

void generate_random_matrix(int *matrix, int n) {
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = rand() % 10; // Random values between 0 and 9
    }
}


int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);

    srand(time(NULL));

    // Allocate host memory
    // For the matrix and returned result c.
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize matrices a and b
    generate_random_matrix(a, MATRIX_SIZE);
    generate_random_matrix(b, MATRIX_SIZE);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);

    // calculate grid size to cover the entire matrix
    dim3 gridSize((MATRIX_SIZE + blockSize.x - 1) / blockSize.x,
                  (MATRIX_SIZE + blockSize.y - 1) / blockSize.y);

    printf("Grid size: (%d, %d)\n", gridSize.x, gridSize.y);
    // Launch the kernel
    matrix_mult<<<gridSize, blockSize>>>(d_a, d_b, d_c, MATRIX_SIZE);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result (for verification)
    printf("Result of matrix multiplication:\n");
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("%d ", c[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}
