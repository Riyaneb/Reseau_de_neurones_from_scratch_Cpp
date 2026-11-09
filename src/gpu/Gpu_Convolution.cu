#include "gpu/GpuCudaCheck.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>

__global__ void forward_im2col_kernel(const Scalar* in, Scalar* out, Index nbPos, Index im2colRow, Index im2colColumn, Index hIn, Index wIn, Index nIn, Index wOut, Index sizeKernel) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < im2colRow && j < im2colColumn) {
        Index caseInCanal = j % (sizeKernel * sizeKernel);
        Index canal = j / (sizeKernel * sizeKernel);
        Index image = i / nbPos;
        Index posKer = i % nbPos;

        Index row_begin = posKer / wOut;
        Index col_begin = posKer % wOut;

        Index hk = caseInCanal / sizeKernel;
        Index wk = caseInCanal % sizeKernel;
        out[i*im2colColumn + j] = in[image * nIn + canal*wIn*hIn + (row_begin + hk) * wIn + (col_begin + wk)];
    }
}

__global__ void forward_output_kernel(const Scalar* in, Scalar* out, Index nbPos, Index nbKer, Index inRow) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    Index k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < inRow && j < nbPos && k < nbKer) {
        out[i * (nbKer * nbPos) + (k*nbPos + j)] = in[(i*nbPos+j) * nbKer + k];
    }
}

__global__ void backward_output_kernel(const Scalar* gradOut, Scalar* in, Index nbPos, Index nbKer, Index inRow) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    Index k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < inRow && j < nbPos && k < nbKer) {
        in[(i*nbPos+j) * nbKer + k] = gradOut[i * (nbKer * nbPos) + (k*nbPos + j)];
    }
}

__global__ void backward_col2im_kernel(const Scalar* gradBlocX, Scalar* gradX, Index nbPos, Index im2colRow, Index im2colColumn, Index hIn, Index wIn, Index nIn, Index wOut, Index sizeKernel) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < im2colRow && j < im2colColumn) {
        Index caseInCanal = j % (sizeKernel * sizeKernel);
        Index canal = j / (sizeKernel * sizeKernel);
        Index image = i / nbPos;
        Index posKer = i % nbPos;

        Index row_begin = posKer / wOut;
        Index col_begin = posKer % wOut;

        Index hk = caseInCanal / sizeKernel;
        Index wk = caseInCanal % sizeKernel;

        Scalar temp = gradBlocX[i * im2colColumn + j];
        atomicAdd(&gradX[image * nIn + canal*wIn*hIn + (row_begin + hk) * wIn + (col_begin + wk)],temp);
    }
}

void forward_im2col_gpu(const Scalar* in, Scalar* out, Index nbPos, Index im2colRow, Index im2colColumn, Index hIn, Index wIn, Index nIn, Index wOut, Index sizeKernel) {
    Index g = 16;
    assert(nbPos > 0 && wOut > 0 && sizeKernel > 0 && "Erreur : nbPos, wOut et sizeKernel doivent être strictement positifs");
    assert((im2colColumn + g - 1) / g <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((im2colRow + g - 1) / g, (im2colColumn + g - 1) / g);
    forward_im2col_kernel<<<blocs, threads>>>(in, out, nbPos, im2colRow, im2colColumn, hIn, wIn, nIn, wOut, sizeKernel);
    CUDA_CHECK(cudaGetLastError());
}

void forward_output_gpu(const Scalar* in, Scalar* out, Index nbPos, Index nbKer, Index inRow) {
    Index g = 8;
    assert((nbPos + g - 1) / g <= 65535 && (nbKer + g - 1) / g <= 65535 && "Erreur : Le nombre de position du kernel et/ou le nombre de kernel dépasse la capacité du calcul par gpu");
    dim3 threads(g,g,g);
    dim3 blocs((inRow + g - 1) / g, (nbPos + g - 1) / g, (nbKer + g - 1) / g);
    forward_output_kernel<<<blocs, threads>>>(in, out, nbPos, nbKer, inRow);
    CUDA_CHECK(cudaGetLastError());
}

void backward_col2im_gpu(const Scalar* gradBlocX, Scalar* gradX, Index nbPos, Index im2colRow, Index im2colColumn, Index hIn, Index wIn, Index nIn, Index wOut, Index sizeKernel) {
    Index g = 16;
    assert(nbPos > 0 && wOut > 0 && sizeKernel > 0 && "Erreur : nbPos, wOut et sizeKernel doivent être strictement positifs");
    assert((im2colColumn + g - 1) / g <= 65535 && "Erreur : Le nombre de colonnes dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((im2colRow + g - 1) / g, (im2colColumn + g - 1) / g);
    backward_col2im_kernel<<<blocs, threads>>>(gradBlocX, gradX, nbPos, im2colRow, im2colColumn, hIn, wIn, nIn, wOut, sizeKernel);
    CUDA_CHECK(cudaGetLastError());
}

void backward_output_gpu(const Scalar* gradOut, Scalar* in, Index nbPos, Index nbKer, Index inRow) {
    Index g = 8;
    assert((nbPos + g - 1) / g <= 65535 && (nbKer + g - 1) / g <= 65535 && "Erreur : Le nombre de position du kernel et/ou le nombre de kernel dépasse la capacité du calcul par gpu");
    dim3 threads(g,g,g);
    dim3 blocs((inRow + g - 1) / g, (nbPos + g - 1) / g, (nbKer + g - 1) / g);
    backward_output_kernel<<<blocs, threads>>>(gradOut, in, nbPos, nbKer, inRow);
    CUDA_CHECK(cudaGetLastError());
}

