#include <algorithm>

#include "gpu/GpuCudaCheck.hpp"
#include "gpu/Memory_manage_matrix.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>

__global__ void gradient_loss_kernel(const Scalar* predic, const Scalar* target, Scalar* result, Index choice, Index row, Index col) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row * col) {
        switch (choice) {
            case 0:
                result[i] = (predic[i] - target[i]) / row;
                break;
            case 1: {
                Scalar y = target[i];
                Scalar p = fminf(fmaxf(predic[i], 1e-7f), 1.0f - 1e-7f);
                result[i] = ((p-y)/(p*(1-p))) / static_cast<Scalar>(row);
                break;
            }
            case 2: {
                Scalar y = target[i];
                Scalar p = fmaxf(predic[i], 1e-7f);
                result[i] = ((-1) * (y/p)) / static_cast<Scalar>(row);
                break;
            }
        }
    }
}

__global__ void calcul_loss_kernel(const Scalar* predic, const Scalar* target, Scalar* result, Index choice, Index row, Index col) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row * col) {
        switch (choice) {
            case 0: {
                Scalar y = target[i];
                Scalar p = predic[i];
                Scalar loss = (y-p) * (y-p);
                atomicAdd(result, loss / (2.0f * static_cast<Scalar>(row)));
                break;
            }
            case 1: {
                Scalar y = target[i];
                Scalar p = fminf(fmaxf(predic[i], 1e-7f), 1.0f - 1e-7f);
                Scalar loss = y * logf(p) + (1-y) * logf(1-p);
                atomicAdd(result, (-1) * loss / static_cast<Scalar>(row));
                break;
            }
            case 2: {
                Scalar y = target[i];
                Scalar p = fmaxf(predic[i], 1e-7f);
                Scalar loss = y * logf(p);
                atomicAdd(result, (-1) * loss / static_cast<Scalar>(row));
                break;
            }

        }
    }
}

void gradient_loss_gpu(const Scalar* predic, const Scalar* target, Scalar* result, Index choice, Index row, Index col) {
    Index threads = 256;
    Index blocs = (threads + row*col -1) / threads;
    gradient_loss_kernel<<<blocs, threads>>>(predic, target, result, choice, row, col);
    CUDA_CHECK(cudaGetLastError());
}

void calcul_loss_gpu(const Scalar* predic, const Scalar* target, Scalar* result, Index choice, Index row, Index col) {
    Index threads = 256;
    Index blocs = (threads + row*col -1) / threads;
    calcul_loss_kernel<<<blocs, threads>>>(predic, target, result, choice, row, col);
    CUDA_CHECK(cudaGetLastError());
}