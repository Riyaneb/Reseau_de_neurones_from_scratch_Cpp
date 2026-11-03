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
    inline std::vector<std::string> &get_names() {
        return layers_name;
    }
    inline std::vector<double> &get_forward_times() {
        return forward_times;
    }
    inline std::vector<double> &get_backward_times() {
        return backward_times;
    }


private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<std::string> layers_name;
    std::vector<double> forward_times;
    std::vector<double> backward_times;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_NETWORK_HPP
