#include "nn/Activations.hpp"
#include <cassert>

Scalar ReLU::apply(Scalar value) const {
    if (value < 0) {
        return 0;
    }
    else {
        return value;
    }
}

Scalar ReLU::derivate_fn(Scalar const &value) const {
    if (value <= 0) {
        return 0;
    }
    else {
        return 1;
    }
}

Matrix ReLU::forward(Matrix const &in) {
    assert(in.get_column() == nbInput && "Erreur : Dimension dans le forward activationLayer Incorrecte");
    input = in;
    Matrix result(input.get_row(), input.get_column());
    #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
    for (Index i = 0; i < input.get_row(); i++) {
        for (Index j = 0; j < input.get_column(); j++) {
            Scalar temp = input.get_value(i,j);
            if (temp > 0) {
                result.get_value(i,j) = temp;
            }
        }
    }
    return result;
}

Matrix ReLU::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == input.get_column() && gradientOut.get_row() == input.get_row() && "Dimension de la matrice des gradients ne correspond pas au nombre de neurones de la couche précédente");
    Matrix result(gradientOut.get_row(), gradientOut.get_column(),0);

    #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
    for (Index i = 0; i < input.get_row(); i++) {
        for (Index j = 0; j < input.get_column(); j++) {
            Scalar temp = input.get_value(i,j);
            if (temp > 0) {
                result.get_value(i,j) = gradientOut.get_value(i,j);
            }
        }
    }
    return result;
}

std::unique_ptr<Layer> ReLU::clone() const {
    return std::make_unique<ReLU>(*this);
}

Scalar Sigmoid::apply(Scalar value) const {
    return 1.0/(1.0 + std::exp(-value));
}

Scalar Sigmoid::derivate_fn(Scalar const &value) const {
    Scalar temp = Sigmoid::apply(value);
    return temp * (1.0 - temp);
}

std::unique_ptr<Layer> Sigmoid::clone() const {
    return std::make_unique<Sigmoid>(*this);
}

Scalar Tanh::apply(Scalar value) const {
    return std::tanh(value);
}

Scalar Tanh::derivate_fn(Scalar const &value) const {
    return 1.0 - pow(std::tanh(value),2);
}

std::unique_ptr<Layer> Tanh::clone() const {
    return std::make_unique<Tanh>(*this);
}

Scalar Identity::apply(Scalar value) const {
    return value;
}

Scalar Identity::derivate_fn(Scalar const &value) const {
    if (value <= 0) {
    }
    return 1;
}

std::unique_ptr<Layer> Identity::clone() const {
    return std::make_unique<Identity>(*this);
}