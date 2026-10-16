#include "nn/Losses.hpp"
#include <algorithm>

Scalar QuadraLoss::calculate_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    Scalar loss_value = 0.0;
    for (Index i = 0; i < prediction.get_row(); i++) {
        for (Index j = 0; j < prediction.get_column(); j++) {
            loss_value += std::pow(target.get_value(i,j) - prediction.get_value(i,j),2);
        }
    }
    loss_value = loss_value / (2.00 * prediction.get_row());
    return loss_value;
}

Matrix QuadraLoss::calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    Matrix grad = prediction - target;
    Scalar n = prediction.get_row();
    for (Index i = 0; i < grad.get_row(); i++) {
        for (Index j = 0; j < grad.get_column(); j++) {
            grad.get_value(i,j) = grad.get_value(i,j) / n;
        }
    }
    return grad;
}


Scalar BCE::calculate_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    Scalar loss_value = 0.0;
    for (int i = 0; i < prediction.get_row(); i++) {
        for (int j = 0; j < prediction.get_column(); j++) {
            Scalar y = target.get_value(i,j);
            Scalar p = std::clamp(prediction.get_value(i,j),1e-12,1 - 1e-12);
            loss_value += y * std::log(p) + (1-y) * std::log(1-p);
        }
    }
    loss_value = (-1) * (loss_value / (prediction.get_row()));
    return loss_value;
}

Matrix BCE::calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    Index n = prediction.get_row();
    Index c = prediction.get_column();
    Matrix grad(n,c);
    for (Index i = 0; i < grad.get_row(); i++) {
        for (Index j = 0; j < grad.get_column(); j++) {
            Scalar y = target.get_value(i,j);
            Scalar p = std::clamp(prediction.get_value(i,j),1e-12,1 - 1e-12);
            grad.get_value(i,j) = (p-y)/(p*(1-p));
            grad.get_value(i,j) = grad.get_value(i,j) / n;
        }
    }
    return grad;
}

Scalar CCE::calculate_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    Scalar loss_value = 0.0;
    for (int i = 0; i < prediction.get_row(); i++) {
        for (int j = 0; j < prediction.get_column(); j++) {
            Scalar y = target.get_value(i,j);
            Scalar p = prediction.get_value(i,j);
            if (p < 1e-12) {
                p = 1e-12;
            }
            loss_value += y * std::log(p);
        }
    }
    loss_value = (-1) * (loss_value / (prediction.get_row()));
    return loss_value;
}

Matrix CCE::calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    Index n = prediction.get_row();
    Index c = prediction.get_column();
    Matrix grad(n,c);
    for (Index i = 0; i < grad.get_row(); i++) {
        for (Index j = 0; j < grad.get_column(); j++) {
            Scalar y = target.get_value(i,j);
            Scalar p = prediction.get_value(i,j);
            if (p < 1e-12) {
                p = 1e-12;
            }
            grad.get_value(i,j) = (-1) * (y/p);
            grad.get_value(i,j) = grad.get_value(i,j) / n;
        }
    }
    return grad;
}
