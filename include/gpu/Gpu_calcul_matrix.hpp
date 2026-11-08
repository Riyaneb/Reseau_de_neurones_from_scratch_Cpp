#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void gpu_matrix_multiply(const Scalar* A, const Scalar* B, Scalar* C, Index a_row, Index a_col, Index b_row, Index b_col);

void plusEqual_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r);
void minusEqual_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r);
void multScalar_gpu(Scalar* ptrA, const Scalar s, Index c, Index r);
void hadamard_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r);
void broadcast_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r);
void transpose_gpu(Scalar* ptrA, Scalar* ptrB, Index c, Index r);
void sum_row_gpu(Scalar* ptrA, Scalar* ptrB, Index c, Index r);
void fill_gpu(Scalar* ptrA, Scalar s, Index r, Index c);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_CALCUL_MATRIX_HPP
