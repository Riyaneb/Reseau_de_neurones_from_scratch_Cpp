#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_ACTIVATIONLAYER_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_ACTIVATIONLAYER_HPP

#include "math/Matrix.hpp"
#include "nn/Layer.hpp"

class ActivationLayer : public Layer {
public:
    ActivationLayer(Index nb) : nbInput(nb), input(1,nb) {}
    virtual ~ActivationLayer() {}
    Index nInput() const override;
    Index nOutput() const override;
    virtual std::string layer_name() const = 0;
    std::vector<Index> save_config() const override {
        std::vector<Index> config(0);
        config.push_back(nbInput);
        return config;
    }
protected:
    Index nbInput;
    Matrix input;
    virtual Scalar apply(Scalar value) const = 0;
    virtual Scalar derivate_fn(Scalar const &value) const = 0;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_ACTIVATIONLAYER_HPP
