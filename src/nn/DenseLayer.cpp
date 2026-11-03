#include "nn/DenseLayer.hpp"
#include <cassert>

DenseLayer::DenseLayer(Index nInput,Index nOutput, Random &gen, Index initChoice) : weight(Matrix(nInput,nOutput)), biais(Matrix(1,nOutput)), inNeural(Matrix(1,nInput)), gradWeight(nInput,nOutput,0), gradBiais(1,nOutput,0) {
    assert((initChoice == 0 || initChoice == 1) && "Erreur : Valeur de initChoice incorrect (Glorot : 0 ; He : 1)");
    if (initChoice==0) {
        init_weight_Glorot(weight, gen);
    }
    else if (initChoice==1) {
        init_weight_He(weight,gen);
    }
}

Index DenseLayer::nInput() const {
    return weight.get_row();
}

Index DenseLayer::nOutput() const {
    return weight.get_column();
}

Matrix DenseLayer::forward(Matrix const &in) {
    assert(in.get_column() == weight.get_row() && "Erreur : Nb de colonne de la matrice d'entrée ne correspondant pas au nombre de ligne de weight");
    inNeural = in;
    Matrix result = (inNeural*weight).broadcast(biais);
    return result;
}

Matrix DenseLayer::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == nOutput() && gradientOut.get_row() == inNeural.get_row() && "Erreur : Problème de dimension avec les gradient reçu des couches de sortie");
    gradWeight = inNeural.transpose() * gradientOut;
    gradBiais = gradientOut.sum_row();
    if (need_grad_input) {
        return gradientOut * weight.transpose();
    }
    return Matrix(1,1);
}

void DenseLayer::set_weight(Matrix const &w) {
    assert(weight.get_row() == w.get_row() && weight.get_column() == w.get_column() && "Erreur : Les matrices weight n'ont pas la même dimension");
    weight = w;
}

void DenseLayer::set_biais(Matrix const &b) {
    assert(biais.get_row() == b.get_row() && biais.get_column() == b.get_column() && "Erreur : Les matrices biais n'ont pas la même dimension");
    biais = b;
}

const Matrix& DenseLayer::get_weight() const{
    return weight;
}

const Matrix& DenseLayer::get_biais() const {
    return biais;
}

const Matrix& DenseLayer::get_gradWeight() const {
    return gradWeight;
}

const Matrix &DenseLayer::get_gradBiais() const {
    return gradBiais;
}

void DenseLayer::set_gradients_zero() {
    gradWeight = Matrix(gradWeight.get_row(),gradWeight.get_column(),0);
    gradBiais = Matrix(gradBiais.get_row(),gradBiais.get_column(),0);
}

PairParameters DenseLayer::get_parameters() {
    PairParameters params;
    std::pair<Matrix*, Matrix*> pair_weight;
    std::pair<Matrix*, Matrix*> pair_biais;

    pair_weight.first = &weight;
    pair_weight.second = &gradWeight;

    pair_biais.first = &biais;
    pair_biais.second = &gradBiais;

    params.push_back(pair_weight);
    params.push_back(pair_biais);

    return params;
}

std::unique_ptr<Layer> DenseLayer::clone() const {
    return std::make_unique<DenseLayer>(*this);
}



