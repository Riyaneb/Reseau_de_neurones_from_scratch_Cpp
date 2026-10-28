#include "nn/SoftmaxLayer.hpp"
#include <cassert>
#include <cmath>


Index SoftmaxLayer::nInput() const {
    return nbInput;
}

Index SoftmaxLayer::nOutput() const {
    return nbInput;
}

Scalar SoftmaxLayer::fn_softmax(Matrix const &M,Index row,Index column) {
    Scalar sum = 0;
    Scalar maxi = M.get_value(row, 0);
    for (Index k = 0; k < M.get_column(); k++) {
        if (M.get_value(row,k) > maxi) {
            maxi = M.get_value(row,k);
        }
    }
    for (Index k = 0; k < M.get_column(); k++) {
        sum += std::exp(M.get_value(row,k) - maxi);
    }
    return (std::exp(M.get_value(row,column) - maxi))/sum;
}

Matrix SoftmaxLayer::forward(Matrix const &in) {
    assert(in.get_column() == nbInput && "Erreur : Dimension dans le forward softmax Incorrecte");
    input = in;
    Matrix result(input.get_row(), input.get_column());
    for (Index i = 0; i < input.get_row(); i++) {
        for (Index j = 0; j < input.get_column(); j++) {
            result.get_value(i,j) = fn_softmax(input,i,j);
        }
    }
    return result;
}

Matrix SoftmaxLayer::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == input.get_column() && gradientOut.get_row() == input.get_row() && "Dimension de la matrice des gradients ne correspond pas au nombre de neurones de la couche précédente");
    Matrix result(input.get_row(), input.get_column(),0);
    for (Index i = 0; i < input.get_row(); i++) {
        Scalar sum_grad_x_softmax = 0;
        for (Index k = 0; k < input.get_column(); ++k) {
            Scalar fn_k = fn_softmax(input, i, k);
            Scalar grad_k = gradientOut.get_value(i, k);
            sum_grad_x_softmax += grad_k * fn_k;
        }
        for (Index j = 0; j < input.get_column(); ++j) {
            Scalar fn_j = fn_softmax(input, i, j);
            Scalar grad_j = gradientOut.get_value(i, j);
            result.get_value(i, j) = fn_j * (grad_j - sum_grad_x_softmax);
        }
    }
    return result;
}
std::unique_ptr<Layer> SoftmaxLayer::clone() const {
    return std::make_unique<SoftmaxLayer>(*this);
}
