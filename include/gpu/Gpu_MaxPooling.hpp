#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_MAXPOOLING_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_MAXPOOLING_HPP

#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void backward_MaxPooling_gpu(Scalar* gradX, const Scalar* gradOut, const Scalar* posMaxim,Index nbImage, Index nIn, Index nOut);
void forward_MaxPooling_gpu(Scalar* posMaxim, const Scalar* in, Scalar* out, Index nbImage, Index nIn, Index nOut, Index hIn, Index wIn, Index hOut, Index wOut, Index sizeKernel);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_MAXPOOLING_HPP
