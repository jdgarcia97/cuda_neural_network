#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


int main(){


cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("SM count: %d\n", prop.multiProcessorCount);
printf("Max threads per block: %d\n",
       prop.maxThreadsPerBlock);


}