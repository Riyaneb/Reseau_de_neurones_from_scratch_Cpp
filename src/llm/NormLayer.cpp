#include "llm/NormLayer.hpp"
#include <cmath>
#include <iostream>
#ifdef USE_CUDA
#include "gpu/Gpu_LLM.hpp"
#endif


void NormLayer::set_gradients_zero() {
    gradWeight.fill(0);
    gradBiais.fill(0);
}

Index NormLayer::nInput() const {
    return dimEmbedding;
}

Index NormLayer::nOutput() const {
    return dimEmbedding;
}

Matrix NormLayer::forward(Matrix const &in) {
    assert(in.get_column() == dimEmbedding && in.get_row() > 0 && "Erreur : Dimension de input incorrect");
    assert(in.get_in_gpu() == weight.get_in_gpu() && in.get_in_gpu() == biais.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    Matrix result(in.get_row(),dimEmbedding,0,in.get_in_gpu());
    NormX = in;
    input = in;
    if (in.get_in_gpu()) {
        #ifdef USE_CUDA
            inv = Matrix(in.get_row(),1,0,true);
            mean = Matrix(in.get_row(),1,0,true);
            forward_normalization_gpu(result.get_gpu_ptr(), in.get_gpu_ptr(), NormX.get_gpu_ptr(), inv.get_gpu_ptr(), mean.get_gpu_ptr(), weight.get_gpu_ptr(), biais.get_gpu_ptr(), in.get_row(), dimEmbedding, epsilon);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif

    }
    else {
        inv = Matrix(in.get_row(),1);
        mean = Matrix(in.get_row(),1);
        #pragma omp parallel for if(result.get_row()*result.get_column() >= MIN_SIZE_OMP) schedule(static)
        for (Index i = 0; i < in.get_row(); i++) {
            Scalar temp_mean = 0;
            Scalar variance = 0;

            for (Index j = 0; j < dimEmbedding; j++) {
                temp_mean += in.get_value(i, j);
            }
            temp_mean /= static_cast<Scalar>(dimEmbedding);
            mean.get_value(i, 0) = temp_mean;

            for (Index j = 0; j < dimEmbedding; j++) {
                Scalar diff = in.get_value(i, j) - temp_mean;
                variance += diff * diff;
            }
            variance /= static_cast<Scalar>(dimEmbedding);

            Scalar inv_temp = 1.0f / std::sqrt(variance + epsilon);
            inv.get_value(i, 0) = inv_temp;

            for (Index j = 0; j < dimEmbedding; j++) {
                Scalar normalize = (in.get_value(i, j) - temp_mean) * inv_temp;
                NormX.get_value(i, j) = normalize;
                result.get_value(i, j) = weight.get_value(0, j) * normalize + biais.get_value(0, j);
            }
        }
    }
    return result;
}

Matrix NormLayer::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_row() > 0 && "Erreur : Dimension de input incorrect");
    assert(gradientOut.get_row() == input.get_row() && gradientOut.get_column() == dimEmbedding && "Erreur : Dimension de gradOut incorrecte");
    assert(gradientOut.get_in_gpu() == input.get_in_gpu() && gradientOut.get_in_gpu() == gradWeight.get_in_gpu() && gradientOut.get_in_gpu() == gradBiais.get_in_gpu() && "Erreur : Les matrices ne sont pas dans le meme device");

    Matrix gradX(input.get_row(),input.get_column(),0,gradientOut.get_in_gpu());

    gradWeight = hadamard(gradientOut, NormX).sum_row();
    gradBiais = gradientOut.sum_row();

    if (gradientOut.get_in_gpu()) {
        #ifdef USE_CUDA
            Matrix mean_gradNorm(gradientOut.get_row(),1,0,true);
            Matrix mean_temp(gradientOut.get_row(),1,0,true);
            backward_normalization_gpu(gradX.get_gpu_ptr(), gradientOut.get_gpu_ptr(), weight.get_gpu_ptr(), NormX.get_gpu_ptr(), mean_gradNorm.get_gpu_ptr(), mean_temp.get_gpu_ptr(), inv.get_gpu_ptr(), gradientOut.get_row(), dimEmbedding);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(gradX.get_row()*gradX.get_column() >= MIN_SIZE_OMP) schedule(static)
        for (Index i = 0; i < gradX.get_row(); i++) {
            Scalar mean_gradNorm = 0;
            Scalar mean_temp = 0;
            for (Index j = 0; j < dimEmbedding; j++) {
                mean_gradNorm += (gradientOut.get_value(i, j) * weight.get_value(0,j)) / static_cast<Scalar>(dimEmbedding);
                mean_temp += (gradientOut.get_value(i, j) * weight.get_value(0,j) * NormX.get_value(i,j)) / static_cast<Scalar>(dimEmbedding);
            }
            for (Index j = 0; j < dimEmbedding; j++) {
                gradX.get_value(i,j) = ((gradientOut.get_value(i, j) * weight.get_value(0,j)) - mean_gradNorm - NormX.get_value(i,j) * mean_temp) * inv.get_value(i,0);
            }
        }
    }
    return gradX;
}

PairParameters NormLayer::get_parameters() {
    PairParameters params;
    std::pair<Matrix*, Matrix*> pair_weight;
    std::pair<Matrix*, Matrix*> pair_pos;

    pair_weight.first = &weight;
    pair_weight.second = &gradWeight;

    pair_pos.first = &biais;
    pair_pos.second = &gradBiais;

    params.push_back(pair_weight);
    params.push_back(pair_pos);

    return params;
}

std::unique_ptr<Layer> NormLayer::clone() const {
    return std::make_unique<NormLayer>(*this);
}