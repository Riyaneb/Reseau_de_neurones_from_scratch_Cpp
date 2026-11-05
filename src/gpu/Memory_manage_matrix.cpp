#include "gpu/Memory_manage_matrix.hpp"
#include <cstdint>
#include <iostream>

bool gpu_init_memory() {
    cudaMemPool_t pool;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "Erreur : Impossible n'obtenir les informations du GPU" << std::endl;
        exit(EXIT_FAILURE);
    }
    uint64_t maxi = prop.totalGlobalMem;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,0));
    CUDA_CHECK(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&maxi));
    return true;
}

Scalar *gpu_alloc(long long sizeMat) {
    static bool call = gpu_init_memory();
    (void)call;
    size_t octets = sizeMat*sizeof(Scalar);
    Scalar* ptr = nullptr;
    CUDA_CHECK(cudaMallocAsync(&ptr, octets, 0));
    return ptr;
}

void gpu_free(Scalar *gpu_ptr) {
    if (gpu_ptr != nullptr) {
        CUDA_CHECK(cudaFreeAsync(gpu_ptr,0));
    }
}
