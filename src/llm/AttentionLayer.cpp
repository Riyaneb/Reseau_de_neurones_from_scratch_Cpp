#include "llm/AttentionLayer.hpp"
#include <iostream>
#include <cstdlib>
#ifdef USE_CUDA
#include "gpu/Gpu_LLM.hpp"
#endif
AttentionLayer::AttentionLayer(Index dim, Index nbH, Random &gen) : dimEmbedding(dim), nbHead(nbH), context_size(0), layerQ(dim,dim,gen,0), layerK(dim,dim,gen,0), layerV(dim,dim,gen,0), layerFinal(dim,dim,gen,0), Q(1,1), K(1,1), V(1,1), attentionMatrix(1,1){
    if (nbH <= 0 || dim % nbH != 0) {
        std::cerr<<"Erreur : La taille du vecteur doit être divisible par le nombre de tête"<<std::endl;
        exit(EXIT_FAILURE);
    }
    column_per_head = dim / nbH;
}

Index AttentionLayer::nInput() const {
    return dimEmbedding;
}

Index AttentionLayer::nOutput() const {
    return dimEmbedding;
}

void AttentionLayer::set_gradients_zero() {
    layerQ.set_gradients_zero();
    layerK.set_gradients_zero();
    layerV.set_gradients_zero();
    layerFinal.set_gradients_zero();
}

PairParameters AttentionLayer::get_parameters() {
    auto parameters1 = layerQ.get_parameters();
    auto parameters2 = layerK.get_parameters();
    auto parameters3 = layerV.get_parameters();
    auto parameters4 = layerFinal.get_parameters();

    PairParameters parametersFinal;
    parametersFinal.reserve(parameters1.size() + parameters2.size() + parameters3.size() + parameters4.size());

    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(parameters1.begin()), std::make_move_iterator(parameters1.end()));
    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(parameters2.begin()), std::make_move_iterator(parameters2.end()));
    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(parameters3.begin()), std::make_move_iterator(parameters3.end()));
    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(parameters4.begin()), std::make_move_iterator(parameters4.end()));

    return parametersFinal;
}

std::unique_ptr<Layer> AttentionLayer::clone() const {
    return std::make_unique<AttentionLayer>(*this);
}

Matrix AttentionLayer::remade_matrix_split_head(Matrix const &m, Index context, Index nb_head) {
    assert(m.get_row()%context == 0 && m.get_column()%nb_head == 0 && "Erreur : Dimension incorrecte");
    Matrix result(m.get_row()*nb_head,m.get_column() / nb_head, 0, m.get_in_gpu());
    if (m.get_in_gpu()) {
        #ifdef USE_CUDA
            split_head_gpu(result.get_gpu_ptr(), m.get_gpu_ptr(), context, nb_head, m.get_row(), m.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        Index nHeadPerCol = m.get_column() / nb_head;
        #pragma omp parallel for if(m.get_row()*m.get_column() >= MIN_SIZE_OMP) schedule(static)
        for (Index i=0; i<m.get_row(); i++) {
            for (Index j=0; j<m.get_column(); j++) {
                Index nbF = i/context;
                Index posF = i%context;
                Index nhead = j/ nHeadPerCol;
                Index colHead = j % nHeadPerCol;
                result.get_value((nbF*nb_head + nhead)*context + posF, colHead) = m.get_value(i,j);
            }
        }
    }
    return result;
}


Matrix AttentionLayer::merge_head(Matrix const &m, Index context, Index nb_head) {
    assert(m.get_row()%(context*nb_head) == 0 && "Erreur : Dimension incorrecte");
    Matrix result(m.get_row() / nb_head,m.get_column() * nb_head, 0, m.get_in_gpu());
    if (m.get_in_gpu()) {
        #ifdef USE_CUDA
            merge_head_gpu(result.get_gpu_ptr(), m.get_gpu_ptr(), context, nb_head, result.get_row(), result.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(m.get_row()*m.get_column() >= MIN_SIZE_OMP) schedule(static)
        for (Index i=0; i<result.get_row(); i++) {
            for (Index j=0; j<result.get_column(); j++) {
                Index nbF = i/context;
                Index posF = i%context;
                Index nhead = j/ m.get_column();
                Index colHead = j % m.get_column();
                result.get_value(i,j) = m.get_value((nbF*nb_head + nhead)*context + posF, colHead);
            }
        }
    }
    return result;
}

Matrix AttentionLayer::softmax_masked(Matrix const &score, Index context){
    assert(context > 0 && "Erreur : La taille de context doit être supérieur à 0");
    assert(context == score.get_column() && "Erreur : Le nombre de colonne du score ne correspond pas à la taille du contexte");
    assert(score.get_row()  context)
}