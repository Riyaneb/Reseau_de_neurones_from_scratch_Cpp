#include <algorithm>

#include "gpu/GpuCudaCheck.hpp"
#include "gpu/Memory_manage_matrix.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>


__global__ void backward_MaxPooling_kernel(Scalar* gradX, const Scalar* gradOut, const Scalar* posMaxim,Index nbImage, Index nIn, Index nOut) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nbImage && j < nOut) {
        gradX[i * nIn + static_cast<Index>(posMaxim[i*nOut+j])] = gradOut[i*nOut+j];
    }
}

__global__ void forward_MaxPooling_kernel(Scalar* posMaxim, const Scalar* in, Scalar* out, Index nbImage, Index nIn, Index nOut, Index hIn, Index wIn, Index hOut, Index wOut, Index sizeKernel) {
    Index i = blockIdx.x * blockDim.x + threadIdx.x;
    Index j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nbImage && j < nOut) {
        Index caseC = hOut * wOut;
        Index canal = j / caseC;
        Index wPos = (j % caseC) % wOut;
        Index hPos = (j % caseC) / wOut;
        Index row = hPos * sizeKernel;
        Index col = wPos * sizeKernel;
        Scalar maxi = in[i*nIn + (canal*hIn*wIn + row*wIn + col)];
        Index maxi_id = canal*hIn*wIn + row*wIn + col;
        for (Index a = 0; a < sizeKernel; a++) {
            for (Index b = 0; b < sizeKernel; b++) {
                Scalar temp = in[i*nIn + (canal * (hIn*wIn) + (row+a)*wIn + (col+b))];
                if (temp > maxi) {
                    maxi = temp;
                    maxi_id = (canal * (hIn*wIn) + (row+a)*wIn + (col+b));
                }
            }
        }
        posMaxim[i*nOut+j] = maxi_id;
        out[i*nOut + j] = maxi;
    }
}

//c*(height*width) + h*width + w

void backward_MaxPooling_gpu(Scalar* gradX, const Scalar* gradOut, const Scalar* posMaxim,Index nbImage, Index nIn, Index nOut) {
    Index g = 16;
    assert((nOut + g - 1) / g <= 65535 && "Erreur : Le nombre de sortie dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((nbImage + g - 1) / g, (nOut + g - 1) / g);
    backward_MaxPooling_kernel<<<blocs, threads>>>(gradX, gradOut, posMaxim, nbImage, nIn, nOut);
    CUDA_CHECK(cudaGetLastError());
}

void forward_MaxPooling_gpu(Scalar* posMaxim, const Scalar* in, Scalar* out, Index nbImage, Index nIn, Index nOut, Index hIn, Index wIn, Index hOut, Index wOut, Index sizeKernel) {
    Index g = 16;
    assert((nOut + g - 1) / g <= 65535 && "Erreur : Le nombre de sortie dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((nbImage + g - 1) / g, (nOut + g - 1) / g);
    forward_MaxPooling_kernel<<<blocs, threads>>>(posMaxim,in,out, nbImage, nIn, nOut, hIn, wIn, hOut, wOut, sizeKernel);
    CUDA_CHECK(cudaGetLastError());
}