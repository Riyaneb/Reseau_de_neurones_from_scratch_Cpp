#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_ACTIVATIONS_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_ACTIVATIONS_HPP

#include "ActivationLayer.hpp"
#include <cmath>

class ReLU : public ActivationLayer {
public:
    ReLU(Index nb) : ActivationLayer(nb) {}
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    std::unique_ptr<Layer> clone() const override;
    std::string layer_name() const override {
        return "ReLU";
    }
protected:
    Scalar apply(Scalar value) const override;
    Scalar derivate_fn(Scalar const &value) const override;
};

class Sigmoid : public ActivationLayer {
public:
    Sigmoid(Index nb) : ActivationLayer(nb) {}
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    std::unique_ptr<Layer> clone() const override;
    std::string layer_name() const override {
        return "Sigmoid";
    }
protected:
    Scalar apply(Scalar value) const override;
    Scalar derivate_fn(Scalar const &value) const override;
};

class Tanh : public ActivationLayer {
public:
    Tanh(Index nb) : ActivationLayer(nb) {}
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    std::unique_ptr<Layer> clone() const override;
    std::string layer_name() const override {
        return "Tanh";
    }
protected:
    Scalar apply(Scalar value) const override;
    Scalar derivate_fn(Scalar const &value) const override;
};

class Identity : public ActivationLayer {
public:
    Identity(Index nb) : ActivationLayer(nb) {}
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    std::unique_ptr<Layer> clone() const override;
    std::string layer_name() const override {
        return "Identity";
    }
protected:
    Scalar apply(Scalar value) const override;
    Scalar derivate_fn(Scalar const &value) const override;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_ACTIVATIONS_HPP
