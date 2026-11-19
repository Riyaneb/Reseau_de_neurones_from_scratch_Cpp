#include "gpu/GpuCudaCheck.hpp"
#include "gpu/Memory_manage_matrix.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>


__global__ void plusEqual_kernel(Scalar* A, const Scalar* B, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        A[i] += B[i];
    }
}

__global__ void minusEqual_kernel(Scalar* A, const Scalar* B, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        A[i] -= B[i];
    }
}

__global__ void multScalar_kernel(Scalar* A, Scalar s, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        A[i] *= s;
    }
}

__global__ void hadamard_kernel(Scalar* A, const Scalar* B, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        A[i] *= B[i];
    }
}

__global__ void broadcast_kernel(Scalar* A, const Scalar* B, Index r, Index c) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < c && y < r) {
        Index i = y*c + x;
        A[i] += B[x];
    }
}

__global__ void transpose_kernel(Scalar* A, const Scalar* B, Index r, Index c) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < c && y < r) {
        Index i = y*c + x;
        Index j = x*r + y;
        A[j] = B[i];
    }
}

__global__ void sum_row_kernel(Scalar* A, const Scalar* B, Index r, Index c) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < c && y < r) {
        Index i = y*c + x;
        atomicAdd(&A[x], B[i]);
    }
}

__global__ void fill_kernel(Scalar* A, const Scalar s, Index r, Index c) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < r*c) {
        A[x] = s;
    }
}

cublasHandle_t get_handle() {
    static cublasHandle_t handle;
    static bool handle_created = false;
    if (!handle_created) {
        CUDA_CHECK_CUBLAS(cublasCreate(&handle));
        handle_created = true;
    }
    return handle;
}

void gpu_matrix_multiply(const Scalar* A, const Scalar* B, Scalar* C, Index a_row, Index a_col, Index b_row, Index b_col) {
    if (a_col != b_row) {
        std::cerr << "Erreur : Le nombre de colonnes de A ne correspond pas au nombre de lignes de B" << std::endl;
        exit(EXIT_FAILURE);
    }
    static cublasHandle_t handle = get_handle();
    const float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, b_col, a_row, a_col, &alpha, B, b_col, A, a_col, &beta, C, b_col));
}

void gpu_batch_matrix_multiply(const Scalar* A, const Scalar* B, Scalar* C, Index a_row, Index a_col, Index b_row, Index b_col, Index nbBloc, bool A_transp, bool B_transp, Scalar coef) {
    assert(nbBloc > 0 && a_row > 0 && a_col > 0 && b_row > 0 && b_col > 0 && "Erreur : Les dimensions et le nombre de blocs doivent être strictement positifs");
    assert(a_row % nbBloc == 0 && b_row % nbBloc == 0 && "Erreur : Le nombre de lignes doit être divisible par le nombre de blocs");
    assert(!(A_transp && B_transp) && "Erreur : Les deux matrices ne peuvent pas être transposées en même temps");
    Index rowBlocA = a_row / nbBloc;
    Index rowBlocB = b_row / nbBloc;
    Index m = A_transp ? nbBloc * a_col : a_row;
    Index n = B_transp ? b_row / nbBloc : b_col;
    Index k = A_transp ? a_row / nbBloc : a_col;

    static cublasHandle_t handle = get_handle();
    auto op1 = B_transp ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto op2 = A_transp ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float beta = 0.0f;
    CUDA_CHECK_CUBLAS(cublasSgemmStridedBatched(handle,op1,op2,n,m/nbBloc,k,&coef,B,b_col,rowBlocB*b_col,A,a_col, rowBlocA*a_col,&beta,C,n,(m/nbBloc)*n,nbBloc));
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

void plusEqual_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r) {
    Index threads = 256;
    Index blocs = (c*r + threads - 1) / threads;
    plusEqual_kernel<<<blocs, threads>>>(ptrA, ptrB, c*r);
    CUDA_CHECK(cudaGetLastError());
}

void minusEqual_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r) {
    Index threads = 256;
    Index blocs = (c*r + threads - 1) / threads;
    minusEqual_kernel<<<blocs, threads>>>(ptrA, ptrB, c*r);
    CUDA_CHECK(cudaGetLastError());
}

void multScalar_gpu(Scalar* ptrA, const Scalar s, Index c, Index r) {
    Index threads = 256;
    Index blocs = (c*r + threads - 1) / threads;
    multScalar_kernel<<<blocs, threads>>>(ptrA, s, c*r);
    CUDA_CHECK(cudaGetLastError());
}

void hadamard_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r) {
    Index threads = 256;
    Index blocs = (c*r + threads - 1) / threads;
    hadamard_kernel<<<blocs, threads>>>(ptrA, ptrB, c*r);
    CUDA_CHECK(cudaGetLastError());
}

void broadcast_gpu(Scalar* ptrA, const Scalar* ptrB, Index c, Index r) {
    Index g = 16;
    assert((r + g - 1) / g <= 65535 && "Erreur : Le nombre de ligne dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((c + g - 1) / g, (r + g - 1) / g);
    broadcast_kernel<<<blocs, threads>>>(ptrA, ptrB, r, c);
    CUDA_CHECK(cudaGetLastError());
}

void transpose_gpu(Scalar* ptrA, Scalar* ptrB, Index c, Index r) {
    Index g = 16;
    assert((r + g - 1) / g <= 65535 && "Erreur : Le nombre de ligne dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((c + g - 1) / g, (r + g - 1) / g);
    transpose_kernel<<<blocs, threads>>>(ptrA, ptrB, r, c);
    CUDA_CHECK(cudaGetLastError());
}

void sum_row_gpu(Scalar* ptrA, Scalar* ptrB, Index c, Index r) {
    Index g = 16;
    assert((r + g - 1) / g <= 65535 && "Erreur : Le nombre de ligne dépasse la capacité du calcul par gpu");
    dim3 threads(g,g);
    dim3 blocs((c + g - 1) / g, (r + g - 1) / g);
    sum_row_kernel<<<blocs, threads>>>(ptrA, ptrB, r, c);
    CUDA_CHECK(cudaGetLastError());
}

void fill_gpu(Scalar* ptrA, Scalar s, Index r, Index c) {
    Index threads = 256;
    Index blocs = (c*r + threads - 1) / threads;
    fill_kernel<<<blocs, threads>>>(ptrA, s, r, c);
    CUDA_CHECK(cudaGetLastError());
}