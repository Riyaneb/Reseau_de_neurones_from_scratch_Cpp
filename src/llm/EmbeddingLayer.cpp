#include "llm/EmbeddingLayer.hpp"
#include "nn/Initializers.hpp"
#include <iostream>
#ifdef USE_CUDA
#include "gpu/Gpu_LLM.hpp"
#endif


EmbeddingLayer::EmbeddingLayer(Index vocSize, Index dimEmbed, Index size_context_max, Random &gen) : vocabSize(vocSize), dimEmbedding(dimEmbed), context_max(size_context_max), actual_context(size_context_max), weight(vocSize, dimEmbed), pos(size_context_max, dimEmbed), gradWeight(vocSize, dimEmbed), gradPos(size_context_max,dimEmbed), input(1,1) {
    init_weight_embedding(weight, gen);
    init_weight_embedding(pos, gen);
}

void EmbeddingLayer::set_gradients_zero() {
    gradWeight.fill(0);
    gradPos.fill(0);
}

Index EmbeddingLayer::nInput() const {
    return 1;
}

Index EmbeddingLayer::nOutput() const {
    return dimEmbedding;
}

Matrix EmbeddingLayer::forward(Matrix const &in) {
    assert(in.get_column() == 1 && in.get_row() > 0 && "Erreur : Dimension de input incorrect");
    assert(actual_context <= context_max && "Erreur : La taille de contexte est supérieur à la limite max du modèle");
    assert(in.get_row()%actual_context==0 && "Erreur : Taille du contexte actuelle invalide pour ce lot de fenêtre");
    Matrix res(in.get_row(), dimEmbedding,0,in.get_in_gpu());
    input = in;
    assert(in.get_in_gpu() == weight.get_in_gpu() && in.get_in_gpu() == pos.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    if (in.get_in_gpu()) {
    #ifdef USE_CUDA
        forward_embedding_gpu(res.get_gpu_ptr(), in.get_gpu_ptr(), weight.get_gpu_ptr(), pos.get_gpu_ptr(), actual_context, res.get_row(), dimEmbedding);
    #else
        std::cerr << "Erreur : CUDA non disponible" << std::endl;
        exit(EXIT_FAILURE);
    #endif
    }
    else {
        #pragma omp parallel for if(res.get_row()*res.get_column() >= MIN_SIZE_OMP) schedule(static)
        for (Index i = 0; i < in.get_row(); i++) {
            Index x = static_cast<Index>(in.get_value(i,0));
            assert(x >= 0 && x < vocabSize && "Erreur : Token invalide");
            for (Index j = 0; j < dimEmbedding; j++) {
                res.get_value(i,j) = weight.get_value(x,j) + pos.get_value(i%actual_context,j);
            }
        }
    }
    return res;
}

Matrix EmbeddingLayer::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_row() > 0 && "Erreur : Dimension de input incorrect");
    assert(gradientOut.get_row() == input.get_row() && gradientOut.get_column() == dimEmbedding && "Erreur : Dimension de gradOut incorrecte");
    assert(gradientOut.get_in_gpu() == input.get_in_gpu() && gradientOut.get_in_gpu() == gradWeight.get_in_gpu() && gradientOut.get_in_gpu() == gradPos.get_in_gpu() && "Erreur : Les matrices ne sont pas dans le meme device");
    if (gradientOut.get_in_gpu()) {
        #ifdef USE_CUDA
            backward_embedding_gpu(gradWeight.get_gpu_ptr(), gradPos.get_gpu_ptr(), gradientOut.get_gpu_ptr(), input.get_gpu_ptr(), gradientOut.get_row(), dimEmbedding, actual_context);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        for (Index i = 0; i < gradientOut.get_row() ; i++) {
            Index x = static_cast<Index>(input.get_value(i,0));
            for (Index j = 0; j < dimEmbedding; j++) {
                gradWeight.get_value(x,j) += gradientOut.get_value(i,j);
                gradPos.get_value(i%actual_context,j) += gradientOut.get_value(i,j);
            }
        }
    }
    return Matrix(1,1);
}

PairParameters EmbeddingLayer::get_parameters() {
    PairParameters params;
    std::pair<Matrix*, Matrix*> pair_weight;
    std::pair<Matrix*, Matrix*> pair_pos;

    pair_weight.first = &weight;
    pair_weight.second = &gradWeight;

    pair_pos.first = &pos;
    pair_pos.second = &gradPos;

    params.push_back(pair_weight);
    params.push_back(pair_pos);

    return params;
}

std::unique_ptr<Layer> EmbeddingLayer::clone() const {
    return std::make_unique<EmbeddingLayer>(*this);
}
