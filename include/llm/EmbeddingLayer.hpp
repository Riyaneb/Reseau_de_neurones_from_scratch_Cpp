#ifndef RESEAUDENEURONEFROMSCRATCH_EMBEDDINGLAYER_HPP
#define RESEAUDENEURONEFROMSCRATCH_EMBEDDINGLAYER_HPP

#include "math/Random.hpp"
#include "nn/Layer.hpp"

class EmbeddingLayer : public Layer {
public:
    EmbeddingLayer(Index vocSize, Index dimEmbed, Index size_context_max ,Random &gen);
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;

    void set_actual_context(Index n) {
        assert(n <= context_max && "Erreur : Le doit être plus petit que la taille maximale de contexte");
        actual_context = n;
    }
    
    std::string layer_name() const override {
        return "Embedding";
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
