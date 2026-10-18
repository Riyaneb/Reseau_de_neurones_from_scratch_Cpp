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

    cublasHandle_t handle;
    CUDA_CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, linkA_B, &alpha, d_B, colB, d_A, linkA_B, &beta, d_C, colB));
    CUDA_CHECK(cudaMemcpy(C.get_ptr_matrix(), d_C, memoryC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK_CUBLAS(cublasDestroy(handle));
}


/*void gpu_matrix_multiply(Matrix const &A, Matrix const &B, Matrix &C) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int warp = prop.warpSize;

    int colA = A.get_column();
    int rowA = A.get_row();

    int colB = B.get_column();
    int rowB = B.get_row();

    int nA = colA * rowA;
    int nB = colB * rowB;
    int nC = colA * rowB;

    size_t memoryA = nA * sizeof(float);
    size_t memoryB = nB * sizeof(float);
    size_t memoryC = nC * sizeof(float);

    float *d_A , *d_B , *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, memoryA));
    CUDA_CHECK(cudaMalloc(&d_B, memoryB));
    CUDA_CHECK(cudaMalloc(&d_C, memoryC));

    CUDA_CHECK(cudaMemcpy(d_A, A.get_ptr_matrix(), memoryA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.get_ptr_matrix(), memoryB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C.get_ptr_matrix(), memoryC, cudaMemcpyHostToDevice));

    dim3 thread(warp,warp);
    dim3 block((warp + rowB - 1)/warp,(warp + colA - 1)/warp);

}*/