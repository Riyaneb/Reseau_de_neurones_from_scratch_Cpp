#include "gpu/GpuCudaCheck.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>

__global__ void forward_embedding_kernel(Scalar* res, const Scalar* input, const Scalar* weight, const Scalar* pos, Index context_size, Index row_res, Index dimEmbedding) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < row_res && j < dimEmbedding) {
        Index x = i*dimEmbedding+j;
        Index utf_value = input[i];
        res[x] = weight[utf_value*dimEmbedding+j] + pos[(i%context_size) * dimEmbedding + j];
    }
}

__global__ void backward_embedding_kernel(Scalar* gradW, Scalar* gradP, const Scalar* gradOut, const Scalar* input, Index gradOut_row, Index dimEmbedding, Index actual_context_size) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < gradOut_row && j < dimEmbedding) {
        Index x = i*dimEmbedding+j;
        Index utf_value = input[i];
        atomicAdd(&gradW[utf_value*dimEmbedding+j], gradOut[x]);
        atomicAdd(&gradP[(i%actual_context_size)*dimEmbedding+j], gradOut[x]);
    }
}

__global__ void forward_inv_kernel(const Scalar* input, Scalar* inv, Scalar* mean, Index input_row, Index dimEmbedding, Scalar epsilon) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < input_row) {
        Scalar variance = 0;
        Scalar temp_mean = 0;
        for (Index j = 0; j < dimEmbedding; j++) {
            temp_mean += input[i*dimEmbedding+j] / static_cast<Scalar>(dimEmbedding);
        }
        mean[i] = temp_mean;
        for (Index j = 0; j < dimEmbedding; j++) {
            variance += (input[i*dimEmbedding+j] - temp_mean) * (input[i*dimEmbedding+j] - temp_mean);
        }
        variance /= static_cast<Scalar>(dimEmbedding);
        inv[i] = 1.0f / (sqrtf(variance + epsilon));
    }
}

__global__ void forward_normalization_kernel(Scalar* result, const Scalar* input, const Scalar* inv, const Scalar* mean, const Scalar* weight, const Scalar* biais, Index input_row, Index dimEmbedding) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < input_row && j < dimEmbedding) {
        Index x = i*dimEmbedding+j;
        result[x] = weight[j] * ((input[x] - mean[i]) * inv[i]) + biais[j];
    }
}

void forward_embedding_gpu(Scalar* res, const Scalar* input, const Scalar* weight, const Scalar* pos, Index context_size, Index row_res, Index dimEmbedding) {
    Index g = 16;
    assert(context_size > 0 && dimEmbedding > 0 && "Erreur : context_size et dimEmbedding doit strictmeent être positif");
    assert((dimEmbedding + g - 1) / g <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((row_res + g - 1) / g, (dimEmbedding + g - 1) / g);
    forward_embedding_kernel<<<blocs, threads>>>(res, input, weight, pos, context_size, row_res, dimEmbedding);
    CUDA_CHECK(cudaGetLastError());
}

void backward_embedding_gpu(Scalar* gradW, Scalar* gradP, const Scalar* gradOut, const Scalar* input, Index gradOut_row, Index dimEmbedding, Index actual_context_size) {
    Index g = 16;
    assert(actual_context_size > 0 && dimEmbedding > 0 && "Erreur : actual_context_size et dimEmbedding doit strictmeent être positif");
    assert((dimEmbedding + g - 1) / g <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((gradOut_row + g - 1) / g, (dimEmbedding + g - 1) / g);
    backward_embedding_kernel<<<blocs, threads>>>(gradW, gradP, gradOut, input, gradOut_row, dimEmbedding, actual_context_size);
    CUDA_CHECK(cudaGetLastError());
}

void forward_normalization_gpu(Scalar* result, const Scalar* input, Scalar* inv, Scalar* mean, const Scalar* weight, const Scalar* biais, Index input_row, Index dimEmbedding, Scalar epsilon) {
    Index g2 = 16;
    assert(dimEmbedding > 0 && "Erreur : dimEmbedding doit strictmeent être positif");
    assert((dimEmbedding + g2 - 1) / g2 <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    Index g1 = 256;
    dim3 threads2(g2,g2);
    Index blocs1 = (input_row + g1 - 1) / g1;
    dim3 blocs2((input_row + g2 - 1) / g2, (dimEmbedding + g2 - 1) / g2);
    forward_inv_kernel<<<blocs1, g1>>>(input, inv, mean, input_row, dimEmbedding, epsilon);
    CUDA_CHECK(cudaGetLastError());
    forward_normalization_kernel<<<blocs2, threads2>>>(result, input, inv, mean, weight, biais, input_row, dimEmbedding);
    CUDA_CHECK(cudaGetLastError());
}