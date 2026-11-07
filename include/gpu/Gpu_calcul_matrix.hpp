#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void gpu_matrix_multiply(Matrix const &A, Matrix const &B, Matrix &C);

void plusEqual_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r);
void minusEqual_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r);
void multScalar_gpu(Scalar* ptrA, const Scalar s, Index c, Index r);
void hadamard_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r);
void broadcast_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r);
void transpose_gpu(Scalar* ptrA, Scalar* ptrB, Index c, Index r);
void sum_row_gpu(Scalar* ptrA, Scalar* ptrB, Index c, Index r);
void fill_gpu(Scalar* ptrA, Scalar s, Index r, Index c);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
