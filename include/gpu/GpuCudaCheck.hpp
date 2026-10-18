#ifndef RESEAUDENEURONEFROMSCRATCH_GPUCALCULMATRIX_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPUCALCULMATRIX_HPP

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <device_launch_parameters.h>

#define CUDA_CHECK(appel) \
    do { \
        cudaError_t err = (appel); \
        if (err != cudaSuccess) { \
            std::fprintf(stderr, "Erreur CUDA : %s\n  fichier %s, ligne %d\n",cudaGetErrorString(err), __FILE__, __LINE__);\
            std::exit(EXIT_FAILURE);\
        }\
    } while (0)

#define CUDA_CHECK_CUBLAS(appel) \
    do { \
        cublasStatus_t err = (appel); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::fprintf(stderr, "Erreur cuBLAS : %s\n  fichier %s, ligne %d\n",cublasGetStatusString(err), __FILE__, __LINE__);\
            std::exit(EXIT_FAILURE);\
        }\
    } while (0)

#endif //RESEAUDENEURONEFROMSCRATCH_GPUCALCULMATRIX_HPP
