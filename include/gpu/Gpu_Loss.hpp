#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_LOSS_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_LOSS_HPP

#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void gradient_loss_gpu(const Scalar* predic, const Scalar* target, Scalar* result, Index choice, Index row, Index col);
void calcul_loss_gpu(const Scalar* predic, const Scalar* target, Scalar* result, Index choice, Index row, Index col);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_LOSS_HPP
