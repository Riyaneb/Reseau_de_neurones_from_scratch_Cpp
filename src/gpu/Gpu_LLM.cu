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

__global__ void forward_normalization_kernel(Scalar* result, Scalar* NormX, const Scalar* input, const Scalar* inv, const Scalar* mean, const Scalar* weight, const Scalar* biais, Index input_row, Index dimEmbedding) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < input_row && j < dimEmbedding) {
        Index x = i*dimEmbedding+j;
        Scalar temp = (input[x] - mean[i]) * inv[i];
        NormX[x] = temp;
        result[x] = weight[j] * temp + biais[j];
    }
}

__global__ void backward_mean_kernel(const Scalar* gradientOut, const Scalar* weight, const Scalar* normal, Scalar* mean_gradNorm, Scalar* mean_temp, Index grad_row, Index dimEmbedding) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < grad_row) {
        Scalar t1 = 0;
        Scalar t2 = 0;
        for (Index j = 0; j < dimEmbedding; j++) {
            t1 += (gradientOut[i*dimEmbedding+j] * weight[j]) / static_cast<Scalar>(dimEmbedding);
            t2 += (gradientOut[i*dimEmbedding+j] * weight[j] * normal[i*dimEmbedding+j]) / static_cast<Scalar>(dimEmbedding);
        }
        mean_gradNorm[i] = t1;
        mean_temp[i] = t2;
    }
}

__global__ void backward_normalization_kernel(Scalar* gradX, const Scalar* gradientOut, const Scalar* weight, const Scalar* normal, const Scalar* mean_gradNorm, const Scalar* mean_temp, const Scalar* inv, Index grad_row, Index dimEmbedding) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < grad_row && j < dimEmbedding) {
        Index x = i*dimEmbedding+j;
        gradX[x] = ((gradientOut[x] * weight[j]) - mean_gradNorm[i] - normal[x] * mean_temp[i]) * inv[i];
    }
}

__global__ void split_head_kernel(Scalar* result, const Scalar* m, Index context, Index nb_head, Index m_row, Index m_col) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m_row && j < m_col) {
        Index nbF = i/context;
        Index posF = i%context;
        Index nHeadPerCol = m_col / nb_head;
        Index nhead = j/ nHeadPerCol;
        Index colHead = j % nHeadPerCol;

        Index a = (nbF*nb_head + nhead)*context + posF;
        result[a*nHeadPerCol + colHead] = m[i*m_col + j];
    }
}

__global__ void merge_head_kernel(Scalar* result, const Scalar* m, Index context, Index nb_head, Index result_row, Index result_col) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < result_row && j < result_col) {
        Index nbF = i/context;
        Index posF = i%context;
        Index nHeadPerCol = result_col / nb_head;
        Index nhead = j/ nHeadPerCol;
        Index colHead = j % nHeadPerCol;

        Index a = (nbF*nb_head + nhead)*context + posF;
        result[i*result_col + j] = m[a*nHeadPerCol + colHead];
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

void forward_normalization_gpu(Scalar* result, const Scalar* input, Scalar* NormX, Scalar* inv, Scalar* mean, const Scalar* weight, const Scalar* biais, Index input_row, Index dimEmbedding, Scalar epsilon) {
    Index g2 = 16;
    assert(dimEmbedding > 0 && "Erreur : dimEmbedding doit strictmeent être positif");
    assert((dimEmbedding + g2 - 1) / g2 <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    Index g1 = 256;
    dim3 threads2(g2,g2);
    Index blocs1 = (input_row + g1 - 1) / g1;
    dim3 blocs2((input_row + g2 - 1) / g2, (dimEmbedding + g2 - 1) / g2);
    forward_inv_kernel<<<blocs1, g1>>>(input, inv, mean, input_row, dimEmbedding, epsilon);
    CUDA_CHECK(cudaGetLastError());
    forward_normalization_kernel<<<blocs2, threads2>>>(result, NormX, input, inv, mean, weight, biais, input_row, dimEmbedding);
    CUDA_CHECK(cudaGetLastError());
}

void backward_normalization_gpu(Scalar* gradX, const Scalar* gradientOut, const Scalar* weight, const Scalar* normal, Scalar* mean_gradNorm, Scalar* mean_temp, const Scalar* inv, Index grad_row, Index dimEmbedding) {
    Index g2 = 16;
    assert(dimEmbedding > 0 && "Erreur : dimEmbedding doit strictmeent être positif");
    assert((dimEmbedding + g2 - 1) / g2 <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    Index g1 = 256;
    dim3 threads2(g2,g2);
    Index blocs1 = (grad_row + g1 - 1) / g1;
    dim3 blocs2((grad_row + g2 - 1) / g2, (dimEmbedding + g2 - 1) / g2);
    backward_mean_kernel<<<blocs1, g1>>>(gradientOut, weight, normal, mean_gradNorm, mean_temp, grad_row, dimEmbedding);
    CUDA_CHECK(cudaGetLastError());
    backward_normalization_kernel<<<blocs2, threads2>>>(gradX, gradientOut, weight, normal, mean_gradNorm, mean_temp, inv, grad_row, dimEmbedding);
    CUDA_CHECK(cudaGetLastError());
}

void split_head_gpu(Scalar* result, const Scalar* m, Index context, Index nb_head, Index m_row, Index m_col) {
    Index g = 16;
    assert(context > 0 && nb_head > 0 && "Erreur : context et nb_head doit strictmeent être positif");
    assert((m_col + g - 1) / g <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((m_row + g - 1) / g, (m_col + g - 1) / g);
    split_head_kernel<<<blocs, threads>>>(result, m, context, nb_head, m_row, m_col);
    CUDA_CHECK(cudaGetLastError());
}
void merge_head_gpu(Scalar* result, const Scalar* m, Index context, Index nb_head, Index result_row, Index result_col) {
    Index g = 16;
    assert(context > 0 && nb_head > 0 && "Erreur : context et nb_head doit strictmeent être positif");
    assert((result_col + g - 1) / g <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((result_row + g - 1) / g, (result_col + g - 1) / g);
    merge_head_kernel<<<blocs, threads>>>(result, m, context, nb_head, result_row, result_col);
    CUDA_CHECK(cudaGetLastError());
}
