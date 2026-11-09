#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_OPTIMIZERS_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_OPTIMIZERS_HPP
#include "Optimizer.hpp"
#include "math/Matrix.hpp"

class SGD : public Optimizer {
public :
    SGD(Scalar a = 0.1) : rate(a)  {}
    void update_parameters(PairParameters &parameters) override;
    std::string optimiser_name() const {
        return "SGD";
    }
private:
    Scalar rate;
};

class Adam : public Optimizer {
public :
    Adam(PairParameters const &pair_parameters, Scalar a = 0.001, Scalar b1 = 0.9, Scalar b2 = 0.999, Scalar eps = 1e-8);
    void update_parameters(PairParameters &parameters) override;
    std::string optimiser_name() const {
        return "Adam";
    }
private:
    Scalar rate;
    Scalar beta1;
    Scalar beta2;
    Scalar epsilon;
    Index cpt;
    std::vector<Matrix> first_moments_params;
    std::vector<Matrix> second_moments_params;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_OPTIMIZERS_HPP
