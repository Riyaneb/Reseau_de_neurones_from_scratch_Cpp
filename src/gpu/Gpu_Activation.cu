#include "gpu/GpuCudaCheck.hpp"
#include "gpu/Memory_manage_matrix.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>

__global__ void forward_activation_gpu_kernel(const Scalar* A, Scalar* output, Index choice, Index row, Index column) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row*column) {
        switch (choice) {
            case 0:
                output[i] = fmaxf(0.0f, A[i]);
                break;
            case 1:
                output[i] = tanhf(A[i]);
                break;
            case 2:
                output[i] = 1.0f/(1.0f + expf(-A[i]));
                break;
            case 3:
                output[i] = A[i];
                break;
        }
    }
}

__global__ void backward_activation_gpu_kernel(const Scalar* A, const Scalar* gradientOut, Scalar* result, Index choice, Index row, Index column) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row*column) {
        switch (choice) {
            case 0:
                result[i] = gradientOut[i] * static_cast<Scalar>(A[i] > 0.0f);
                break;
            case 1: {
                Scalar fn = tanhf(A[i]);
                result[i] = gradientOut[i] * (1.0f - fn * fn);
                break;
            }
            case 2: {
                Scalar temp = 1.0f/(1.0f + expf(-A[i]));
                result[i] = gradientOut[i] * temp * (1.0f - temp);
                break;
            }
            case 3:
                result[i] = gradientOut[i] * 1.0f;
                break;
        }
    }
}

__global__ void forward_softmax_gpu_kernel(const Scalar* A, Scalar* output, Index row, Index column) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row) {
        Scalar sum = 0;
        Scalar maxi = A[i*column];
        for (int k = i*column; k < i*column+column; ++k) {
            if (A[k] > maxi) {
                maxi = A[k];
            }
        }
        for (int k = i*column; k < i*column+column; ++k) {
            Scalar fn = expf(A[k] - maxi);
            sum += fn;
            output[k] = fn;
        }
        for (int k = i*column; k < i*column+column; ++k) {
            output[k] /= sum;
        }
    }
}

__global__ void backward_softmax_gpu_kernel(const Scalar* gradientOut, const Scalar* output, Scalar* result, Index row, Index column) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row) {
        Scalar s = 0;
        for (int k = i*column; k < i*column+column; ++k) {
            s += gradientOut[k] * output[k];
        }
        for (int k = i*column; k < i*column+column; ++k) {
            result[k] = output[k] * (gradientOut[k] - s);
        }
    }
}

void forward_activation_gpu(const Scalar* in, Scalar* result, Index choice, Index row, Index column) {
    Index threads = 256;
    Index blocs = (row*column + threads - 1) / threads;
    forward_activation_gpu_kernel<<<blocs, threads>>>(in, result, choice, row, column);
    CUDA_CHECK(cudaGetLastError());
}

void backward_activation_gpu(const Scalar* output, const Scalar* gradientOut, Scalar* result, Index choice, Index row, Index column) {
    Index threads = 256;
    Index blocs = (row*column + threads - 1) / threads;
    backward_activation_gpu_kernel<<<blocs, threads>>>(output, gradientOut, result, choice, row, column);
    CUDA_CHECK(cudaGetLastError());
}

void forward_softmax_gpu(const Scalar* input, Scalar* output, Index row, Index column) {
    Index threads = 256;
    Index blocs = (row + threads - 1) / threads;
    forward_softmax_gpu_kernel<<<blocs, threads>>>(input, output, row, column);
    CUDA_CHECK(cudaGetLastError());
}

void backward_softmax_gpu(const Scalar* gradientOut, const Scalar* output, Scalar* result, Index row, Index column) {
    Index threads = 256;
    Index blocs = (row + threads - 1) / threads;
    backward_softmax_gpu_kernel<<<blocs, threads>>>(gradientOut, output, result, row, column);
    CUDA_CHECK(cudaGetLastError());
}