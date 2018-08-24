#ifndef CUDA_MACROS_H_
#define CUDA_MACROS_H_

#include <iostream>

#define EQ(a, b) \
  ((a) == (b))

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
	    i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK \
  if (cudaSuccess != cudaPeekAtLastError()) \
    	std::cout << "Cuda kernel failed. Error: " \
		<< cudaGetErrorString(cudaPeekAtLastError())

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) std::cout << cudaGetErrorString(error); \
} while (0)

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#endif //CUDA_MACROS_H_
