#ifndef RESEAUDENEURONEFROMSCRATCH_TRANSFORMER_HPP
#define RESEAUDENEURONEFROMSCRATCH_TRANSFORMER_HPP
#include "nn/Layer.hpp"
#include "llm/NormLayer.hpp"
#include "llm/AttentionLayer.hpp"
#include "nn/Activations.hpp"
#include "nn/DenseLayer.hpp"
#include <cassert>


class TransformerBloc : public Layer {
public:
    TransformerBloc(Index d, Index h, Random &gen);
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    void set_gradients_zero() override;
    Index nInput() const override;
    Index nOutput() const override;
    PairParameters get_parameters() override;
    std::unique_ptr<Layer> clone() const override;

    void set_actual_context(Index n) override {
        assert(n > 0 && "Erreur : Le nombre doit être supérieur à 0");
        context = n;
        attention.set_actual_context(n);
    }

    std::string layer_name() const override {
        return "TransformerBloc";
    }

    std::vector<Index> save_config() const override {
        std::vector<Index> config(0);
        config.push_back(dimEmbedding);
        config.push_back(nHead);
        return config;
    }
private:
    Index dimEmbedding;
    Index nHead;
    Index context;
    NormLayer norm1;
    AttentionLayer attention;
    NormLayer norm2;
    DenseLayer dense1;
    GeLU geLu;
    DenseLayer dense2;

};

#endif //RESEAUDENEURONEFROMSCRATCH_TRANSFORMER_HPP
