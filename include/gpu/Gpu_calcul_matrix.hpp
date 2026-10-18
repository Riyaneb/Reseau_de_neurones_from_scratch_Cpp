#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void gpu_matrix_multiply(Matrix const &A, Matrix const &B, Matrix &C);


#endif //RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
