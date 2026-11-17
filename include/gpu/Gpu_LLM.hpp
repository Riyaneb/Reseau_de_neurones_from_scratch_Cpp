#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_LLM_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_LLM_HPP

#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void forward_embedding_gpu(Scalar* res, const Scalar* input, const Scalar* weight, const Scalar* pos, Index context_size, Index row_res, Index dimEmbedding);
void backward_embedding_gpu(Scalar* gradW, Scalar* gradP, const Scalar* gradOut, const Scalar* input, Index gradOut_row, Index dimEmbedding, Index actual_context_size);
void forward_normalization_gpu(Scalar* result, const Scalar* input, Scalar* NormX, Scalar* inv, Scalar* mean, const Scalar* weight, const Scalar* biais, Index input_row, Index dimEmbedding, Scalar epsilon);
void backward_normalization_gpu(Scalar* gradX, const Scalar* gradientOut, const Scalar* weight, const Scalar* normal, Scalar* mean_gradNorm, Scalar* mean_temp, const Scalar* inv, Index grad_row, Index dimEmbedding);
void split_head_gpu(Scalar* result, const Scalar* m, Index context, Index nb_head, Index m_row, Index m_col);
void merge_head_gpu(Scalar* result, const Scalar* m, Index context, Index nb_head, Index result_row, Index result_col);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_LLM_HPP
