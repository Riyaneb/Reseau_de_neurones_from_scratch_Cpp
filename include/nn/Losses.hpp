#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSSES_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSSES_HPP
#include "Loss.hpp"
#include <cassert>
#include <cmath>

class QuadraLoss : public Loss {
public:
    Scalar calculate_loss(Matrix const &prediction, Matrix const &target) const;
    Matrix calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const;
    inline std::string loss_name() const override {
        return "QuadraLoss";
    }
};

class BCE : public Loss {
public:
    Scalar calculate_loss(Matrix const &prediction, Matrix const &target) const;
    Matrix calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const;
    inline std::string loss_name() const override {
        return "BCE";
    }
};

class CCE : public Loss {
public:
    Scalar calculate_loss(Matrix const &prediction, Matrix const &target) const;
    Matrix calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const;
    inline std::string loss_name() const override {
        return "CCE";
    }
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSSES_HPP
