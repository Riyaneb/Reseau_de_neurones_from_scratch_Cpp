#ifndef RESEAUDENEURONEFROMSCRATCH_MEMORY_MANAGE_MATRIX_HPP
#define RESEAUDENEURONEFROMSCRATCH_MEMORY_MANAGE_MATRIX_HPP

#include "gpu/GpuCudaCheck.hpp"
#include "math/Types.hpp"

bool gpu_init_memory();
Scalar* gpu_alloc(long long sizeMat);
void gpu_free(Scalar* gpu_ptr);

#endif //RESEAUDENEURONEFROMSCRATCH_MEMORY_MANAGE_MATRIX_HPP