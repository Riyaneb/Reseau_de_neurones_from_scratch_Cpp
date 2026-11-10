#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_DENSELAYER_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_DENSELAYER_HPP

#include "math/Matrix.hpp"
#include "math/Random.hpp"
#include "nn/Initializers.hpp"
#include "nn/Layer.hpp"

class DenseLayer : public Layer {
public:
    DenseLayer(Index nInput,Index nOutput, Random &gen, Index initChoice=0);
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    Index nInput() const override;
    Index nOutput() const override;
    PairParameters get_parameters() override;
    void set_gradients_zero() override;
    std::unique_ptr<Layer> clone() const override;

    void set_weight(Matrix const &w);
    void set_biais(Matrix const &b);
    const Matrix& get_weight() const;
    const Matrix& get_biais() const;
    const Matrix& get_gradWeight() const;
    const Matrix& get_gradBiais() const;

    std::string layer_name() const override {
        return "DenseLayer";
    }

    std::vector<Index> save_config() const override {
        std::vector<Index> config(0);
        config.push_back(weight.get_row());
        config.push_back(weight.get_column());
        return config;
    }

private:
    Matrix weight;
    Matrix biais;
    Matrix inNeural;
    Matrix gradWeight;
    Matrix gradBiais;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_DENSELAYER_HPP
