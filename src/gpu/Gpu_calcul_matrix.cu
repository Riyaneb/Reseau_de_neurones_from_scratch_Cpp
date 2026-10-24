#include "gpu/GpuCudaCheck.hpp"
#include "math/Matrix.hpp"
#include <cassert>




void gpu_matrix_multiply(Matrix const &A, Matrix const &B, Matrix &C) {
    int rowA = A.get_row();
    int colB = B.get_column();
    int linkA_B = A.get_column();
    assert(A.get_column() == B.get_row() && "Erreur : Le nombre de colonnes de A ne correspond pas au nombre de lignes de B");
    assert(C.get_row() == A.get_row() && C.get_column() == B.get_column() && "Dimension de C incorrecte");
    int nA = rowA * linkA_B;
    int nB = linkA_B * colB;
    int nC = rowA * colB;

    size_t memoryA = nA * sizeof(float);
    size_t memoryB = nB * sizeof(float);
    size_t memoryC = nC * sizeof(float);

    float *d_A , *d_B , *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, memoryA));
    CUDA_CHECK(cudaMalloc(&d_B, memoryB));
    CUDA_CHECK(cudaMalloc(&d_C, memoryC));

    CUDA_CHECK(cudaMemcpy(d_A, A.get_ptr_matrix(), memoryA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.get_ptr_matrix(), memoryB, cudaMemcpyHostToDevice));

    static cublasHandle_t handle;
    static bool handle_created = false;
    if (!handle_created) {
        CUDA_CHECK_CUBLAS(cublasCreate(&handle));
        handle_created = true;
    }
    const float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, linkA_B, &alpha, d_B, colB, d_A, linkA_B, &beta, d_C, colB));
    CUDA_CHECK(cudaMemcpy(C.get_ptr_matrix(), d_C, memoryC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

bool gpu_available() {
    static int nb = 0;
    static bool nb_device_verif = false;
    static bool gpu_avail = false;
    if (!nb_device_verif) {
        cudaGetDeviceCount(&nb);
        if (nb != 0) {
            gpu_avail = true;
        }
        nb_device_verif = true;
    }
   return gpu_avail;
}