#include "gpu/GpuCudaCheck.hpp"
#include "gpu/Memory_manage_matrix.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>

__global__ void Adam_kernel(Scalar* params, const Scalar* first_moment, const Scalar* second_moment, Scalar rate, Scalar epsilon, Index row, Index col) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row * col) {
        params[i] -= (rate * first_moment[i]) / (sqrtf(second_moment[i]) + epsilon);
    }
}

void Adam_gpu(Scalar* params, const Scalar* first_moment, const Scalar* second_moment, Scalar rate, Scalar epsilon, Index row, Index col) {
    Index threads = 256;
    Index blocs = (threads + row*col - 1) / threads;
    Adam_kernel<<<blocs,threads>>>(params, first_moment, second_moment, rate, epsilon, row,col);
    CUDA_CHECK(cudaGetLastError());
}