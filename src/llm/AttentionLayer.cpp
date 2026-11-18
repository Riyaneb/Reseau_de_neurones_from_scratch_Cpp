#include "llm/AttentionLayer.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

#ifdef USE_CUDA
#include "gpu/Gpu_LLM.hpp"
#include "gpu/Gpu_Activation.hpp"
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

Matrix AttentionLayer::forward_softmax_masked(Matrix const &score, Index context){
    assert(context > 0 && "Erreur : La taille de context doit être supérieur à 0");
    assert(context == score.get_column() && "Erreur : Le nombre de colonne du score ne correspond pas à la taille du contexte");
    assert(score.get_row() % context == 0 && "Erreur : Le nombre de ligne doit être divisible par la taille du contexte");
    Matrix res(score.get_row(),score.get_column(),0,score.get_in_gpu());
    if (res.get_in_gpu()) {
        #ifdef USE_CUDA
            forward_softmax_gpu(score.get_gpu_ptr(), res.get_gpu_ptr(), score.get_row(), score.get_column(), context);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(score.get_row()*score.get_column() >= MIN_SIZE_OMP) schedule(static)
        for (Index i=0; i<res.get_row(); i++) {
            Index posTok = i%context;
            Scalar maxi = score.get_value(i,0);
            for (Index j=0; j<=posTok; j++) {
                Scalar temp = score.get_value(i,j);
                if (temp > maxi) {
                    maxi = temp;
                }
            }
            Scalar sum = 0;
            for (Index j=0; j<=posTok; j++) {
                Scalar temp = std::exp(score.get_value(i,j) - maxi);
                sum += temp;
                res.get_value(i,j) = temp;
            }
            for (Index j=0; j<=posTok; j++) {
                res.get_value(i,j) /= sum;
            }
        }
    }
    return res;
}

Matrix AttentionLayer::backward_softmax_masked(Matrix const &grad,Matrix const &attentionMatrix) {
    assert(grad.get_column() == attentionMatrix.get_column() && grad.get_row() == attentionMatrix.get_row() && "Dimension incorrecte");
    assert(grad.get_in_gpu() == attentionMatrix.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    Matrix result(attentionMatrix.get_row(), attentionMatrix.get_column(),0, grad.get_in_gpu());
    if (grad.get_in_gpu()) {
#ifdef USE_CUDA
        backward_softmax_gpu(grad.get_gpu_ptr(), attentionMatrix.get_gpu_ptr(), result.get_gpu_ptr(), attentionMatrix.get_row(), attentionMatrix.get_column());
#else
        std::cerr << "Erreur : CUDA non disponible" << std::endl;
        exit(EXIT_FAILURE);
#endif
    }
    else {
        #pragma omp parallel for if(attentionMatrix.get_row() * attentionMatrix.get_column() > MIN_SIZE_OMP) schedule(static)
        for (Index i = 0; i < attentionMatrix.get_row(); i++) {
            Scalar sum_grad_x_softmax = 0;
            for (Index k = 0; k < attentionMatrix.get_column(); ++k) {
                Scalar fn_k = attentionMatrix.get_value(i, k);
                Scalar grad_k = grad.get_value(i, k);
                sum_grad_x_softmax += grad_k * fn_k;
            }
            for (Index j = 0; j < attentionMatrix.get_column(); ++j) {
                Scalar fn_j = attentionMatrix.get_value(i, j);
                Scalar grad_j = grad.get_value(i, j);
                result.get_value(i, j) = fn_j * (grad_j - sum_grad_x_softmax);
            }
        }
    }
    return result;

}

Matrix AttentionLayer::forward(Matrix const &in) {
    assert(context_size > 0 && "Erreur : Taille du context incorrecte");
    assert(in.get_column() == dimEmbedding && in.get_row() % context_size == 0 && "Erreur : Dimension de in incorrecte");

    Q = remade_matrix_split_head(layerQ.forward(in),context_size,nbHead);
    K = remade_matrix_split_head(layerK.forward(in),context_size,nbHead);
    V = remade_matrix_split_head(layerV.forward(in),context_size,nbHead);

    Scalar coef = 1.0f / std::sqrtf(static_cast<Scalar>(column_per_head));
    Matrix score = Matrix::multiplyMatrixBatch(Q,K, Q.get_row()/context_size,false,true, coef);
    attentionMatrix = forward_softmax_masked(score,context_size);
    Matrix fin = Matrix::multiplyMatrixBatch(attentionMatrix,V,Q.get_row()/context_size,false,false, 1.0f);
    return layerFinal.forward(merge_head(fin,context_size,nbHead));
}

Matrix AttentionLayer::backward(Matrix const &gradientOut) {
    assert(context_size > 0 && "Erreur : Taille du context incorrecte");
    assert(gradientOut.get_column() == dimEmbedding && gradientOut.get_row() % context_size == 0 && "Erreur : Dimension de gradientOut incorrecte");
    Index nbBlocs = Q.get_row() / context_size;
    Scalar coef = 1.0f / std::sqrtf(static_cast<Scalar>(column_per_head));

    Matrix gradFin = remade_matrix_split_head(layerFinal.backward(gradientOut), context_size,nbHead);
    Matrix gradA = Matrix::multiplyMatrixBatch(gradFin,V,nbBlocs,false,true, 1);
    Matrix gradV = Matrix::multiplyMatrixBatch(attentionMatrix, gradFin, nbBlocs, true, false, 1);
    Matrix gradScore = backward_softmax_masked(gradA, attentionMatrix);
    Matrix gradQ = Matrix::multiplyMatrixBatch(gradScore, K, nbBlocs, false, false, coef);
    Matrix gradK = Matrix::multiplyMatrixBatch(gradScore, Q, nbBlocs, true, false, coef);

    return layerQ.backward(merge_head(gradQ,context_size,nbHead)) + layerK.backward(merge_head(gradK,context_size,nbHead)) + layerV.backward(merge_head(gradV,context_size,nbHead));
}