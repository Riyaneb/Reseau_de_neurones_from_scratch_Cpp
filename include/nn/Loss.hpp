#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSS_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSS_HPP
#include "math/Matrix.hpp"

class Loss {
public :
    virtual ~Loss() {}
    virtual Scalar calculate_loss(Matrix const &prediction, Matrix const &target) const = 0;
    virtual Matrix calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const = 0;
    virtual std::string loss_name() const = 0;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_LOSS_HPP
