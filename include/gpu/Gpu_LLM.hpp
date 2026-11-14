#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_LLM_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_LLM_HPP

#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void forward_embedding_gpu(Scalar* res, const Scalar* input, const Scalar* weight, const Scalar* pos, Index context_size, Index row_res, Index dimEmbedding);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_LLM_HPP
