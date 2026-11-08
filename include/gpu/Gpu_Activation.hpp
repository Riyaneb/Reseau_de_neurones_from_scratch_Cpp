#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_ACTIVATION_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_ACTIVATION_HPP

#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void forward_activation_gpu(const Scalar* in, Scalar* result, Index choice, Index row, Index column);
void backward_activation_gpu(const Scalar* output, const Scalar* gradientOut, Scalar* result, Index choice, Index row, Index column);
void forward_softmax_gpu(const Scalar* input, Scalar* output, Index row, Index column);
void backward_softmax_gpu(const Scalar* gradientOut, const Scalar* output, Scalar* result, Index row, Index column);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_ACTIVATION_HPP
