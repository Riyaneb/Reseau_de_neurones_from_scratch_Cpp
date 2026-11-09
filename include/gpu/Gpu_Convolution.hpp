#ifndef RESEAUDENEURONEFROMSCRATCH_GPU_CONVOLUTION_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPU_CONVOLUTION_HPP

#include "math/Matrix.hpp"
#include "gpu/GpuCudaCheck.hpp"

void forward_im2col_gpu(const Scalar* in, Scalar* out, Index nbPos, Index im2colRow, Index im2colColumn, Index hIn, Index wIn, Index nIn, Index wOut, Index sizeKernel);
void forward_output_gpu(const Scalar* in, Scalar* out, Index nbPos, Index nbKer, Index inRow);
void backward_col2im_gpu(const Scalar* gradBlocX, Scalar* gradX, Index nbPos, Index im2colRow, Index im2colColumn, Index hIn, Index wIn, Index nIn, Index wOut, Index sizeKernel);
void backward_output_gpu(const Scalar* gradOut, Scalar* in, Index nbPos, Index nbKer, Index inRow);

#endif //RESEAUDENEURONEFROMSCRATCH_GPU_CONVOLUTION_HPP
