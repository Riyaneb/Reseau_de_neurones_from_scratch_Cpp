#include "nn/ActivationLayer.hpp"
#include <cassert>
#define MIN_SIZE_OMP 32768

Index ActivationLayer::nInput() const {
    return nbInput;
}

Index ActivationLayer::nOutput() const {
    return nbInput;
}

Matrix ActivationLayer::forward(Matrix const &in) {
    assert(in.get_column() == nbInput && "Erreur : Dimension dans le forward activationLayer Incorrecte");
    input = in;
    Matrix result(input.get_row(), input.get_column());
    #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
    for (Index i = 0; i < input.get_row(); i++) {
        for (Index j = 0; j < input.get_column(); j++) {
            result.get_value(i,j) = apply(input.get_value(i,j));
        }
    }
    return result;
}

Matrix ActivationLayer::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == input.get_column() && gradientOut.get_row() == input.get_row() && "Dimension de la matrice des gradients ne correspond pas au nombre de neurones de la couche précédente");
    Matrix gradient_activation_fn(input.get_row(), input.get_column(),0);

    #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
    for (Index i = 0; i < input.get_row(); i++) {
        for (Index j = 0; j < input.get_column(); j++) {
            gradient_activation_fn.get_value(i,j) = derivate_fn(input.get_value(i,j));
        }
    }
    return hadamard(gradientOut,gradient_activation_fn);
}