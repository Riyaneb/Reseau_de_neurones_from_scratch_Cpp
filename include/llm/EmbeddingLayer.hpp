#ifndef RESEAUDENEURONEFROMSCRATCH_EMBEDDINGLAYER_HPP
#define RESEAUDENEURONEFROMSCRATCH_EMBEDDINGLAYER_HPP

#include "math/Random.hpp"
#include "nn/Layer.hpp"

class EmbeddingLayer : public Layer {
public:
    EmbeddingLayer(Index vocSize, Index dimEmbed, Index size_context_max ,Random &gen);
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    void set_gradients_zero() override;
    Index nInput() const override;
    Index nOutput() const override;
    PairParameters get_parameters() override;
    std::unique_ptr<Layer> clone() const override;

    void set_actual_context(Index n) override {
        assert(n <= context_max && n > 0 && "Erreur : Le doit être plus petit que la taille maximale de contexte et supérieur à 0");
        actual_context = n;
    }

    std::string layer_name() const override {
        return "Embedding";
    }
    std::vector<Index> save_config() const override {
        std::vector<Index> config(0);
        config.push_back(vocabSize);
        config.push_back(dimEmbedding);
        config.push_back(context_max);
        return config;
    }

private:
    Index vocabSize;
    Index dimEmbedding;
    Index context_max;
    Index actual_context;

    Matrix weight;
    Matrix pos;
    Matrix gradWeight;
    Matrix gradPos;

    Matrix input;
};

#endif //RESEAUDENEURONEFROMSCRATCH_EMBEDDINGLAYER_HPP
