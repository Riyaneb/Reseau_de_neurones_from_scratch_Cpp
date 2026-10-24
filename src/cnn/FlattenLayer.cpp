#include "cnn/FlattenLayer.hpp"
#include <cassert>

Index FlattenLayer::nInput() const {
    return shape.total_values();
}

Index FlattenLayer::nOutput() const {
    return shape.total_values();
}

Matrix FlattenLayer::forward(Matrix const &in) {
    assert(in.get_column() == nInput() && "Erreur dans les dimension de la matrice d'entrée");
    return in;
}

Matrix FlattenLayer::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == nOutput() && "Erreur dans les dimension de la matrice de gradient");
    return gradientOut;
}

std::unique_ptr<Layer> FlattenLayer::clone() const {
    return std::make_unique<FlattenLayer>(*this);
}
