#ifndef RESEAUDENEURONEFROMSCRATCH_NORMLAYER_HPP
#define RESEAUDENEURONEFROMSCRATCH_NORMLAYER_HPP
#include "nn/Layer.hpp"
#include "math/Matrix.hpp"

class NormLayer : public Layer {
public:
    NormLayer(Index dim) : dimEmbedding(dim), weight(1,dim,1), biais(1,dim,0), gradWeight(1,dim,0), gradBiais(1,dim,0), mean(1,1), inv(1,1),input(1,1), NormX(1,1) {}
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    void set_gradients_zero() override;
    Index nInput() const override;
    Index nOutput() const override;
    PairParameters get_parameters() override;
    std::unique_ptr<Layer> clone() const override;

    std::string layer_name() const override {
        return "NormalizationLayerLLM";
    }
    std::vector<Index> save_config() const override {
        std::vector<Index> config(0);
        config.push_back(dimEmbedding);
        return config;
    }

private:
    Index dimEmbedding;

    Matrix weight;
    Matrix biais;
    Matrix gradWeight;
    Matrix gradBiais;
    Matrix mean;
    Matrix inv;

    Matrix input;
    Matrix NormX;
    static constexpr Scalar epsilon = 1e-5f;
};

#endif //RESEAUDENEURONEFROMSCRATCH_NORMLAYER_HPP
