#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_NETWORK_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_NETWORK_HPP

#include "nn/Layer.hpp"
#include <memory>
#include <vector>
#include <cassert>
#include <utility>

class Network {
public:
    void add(std::unique_ptr<Layer>);
    Matrix forward(Matrix const &X);
    Matrix backward(Matrix const &gradLoss);
    PairParameters get_parameters();
    void set_gradients_zero();
    Network clone() const;
    void to_cpu();
    void to_gpu();
    bool in_gpu();
    std::vector<std::string> &get_names() {
        return layers_name;
    }
    std::vector<std::unique_ptr<Layer>> &get_layers() {
        return layers;
    }


private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<std::string> layers_name;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_NETWORK_HPP
