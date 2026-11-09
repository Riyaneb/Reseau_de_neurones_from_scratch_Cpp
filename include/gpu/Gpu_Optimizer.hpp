#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_OPTIMIZER_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_OPTIMIZER_HPP

#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void Adam_gpu(Scalar* params, const Scalar* first_moment, const Scalar* second_moment, Scalar rate, Scalar epsilon, Index row, Index col);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_OPTIMIZER_HPP
