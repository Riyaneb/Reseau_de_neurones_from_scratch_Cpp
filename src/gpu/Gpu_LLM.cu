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


void forward_embedding_gpu(Scalar* res, const Scalar* input, const Scalar* weight, const Scalar* pos, Index context_size, Index row_res, Index dimEmbedding) {
    Index g = 16;
    assert(context_size > 0 && dimEmbedding > 0 && "Erreur : context_size et dimEmbedding doit strictmeent être positif");
    assert((dimEmbedding + g - 1) / g <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((row_res + g - 1) / g, (dimEmbedding + g - 1) / g);
    forward_embedding_kernel<<<blocs, threads>>>(res, input, weight, pos, context_size, row_res, dimEmbedding);
    CUDA_CHECK(cudaGetLastError());
}