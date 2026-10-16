#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSSES_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSSES_HPP
#include "Loss.hpp"
#include <cassert>
#include <cmath>

class QuadraLoss : public Loss {
public:
    Scalar calculate_loss(Matrix const &prediction, Matrix const &target) const;
    Matrix calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const;
};

class BCE : public Loss {
public:
    Scalar calculate_loss(Matrix const &prediction, Matrix const &target) const;
    Matrix calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const;
};

class CCE : public Loss {
public:
    Scalar calculate_loss(Matrix const &prediction, Matrix const &target) const;
    Matrix calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSSES_HPP
