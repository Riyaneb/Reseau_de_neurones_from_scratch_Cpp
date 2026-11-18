#ifndef RESEAUDENEURONEFROMSCRATCH_ATTENTIONLAYER_HPP
#define RESEAUDENEURONEFROMSCRATCH_ATTENTIONLAYER_HPP
#include "nn/DenseLayer.hpp"
#include "nn/Layer.hpp"

class AttentionLayer : public Layer {
public:
    AttentionLayer(Index dim, Index nbH, Random &gen);
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    void set_gradients_zero() override;
    Index nInput() const override;
    Index nOutput() const override;
    PairParameters get_parameters() override;
    std::unique_ptr<Layer> clone() const override;

    void set_actual_context(Index n) {
        assert(n > 0 && "Erreur : Le nombre doit être supérieur à 0");
        context_size = n;
    }

    std::string layer_name() const override {
        return "AttentionLayer";
    }
    std::vector<Index> save_config() const override {
        std::vector<Index> config(0);
        config.push_back(dimEmbedding);
        config.push_back(nbHead);
        return config;
    }

private:
    Index dimEmbedding;
    Index nbHead;
    Index column_per_head;
    Index context_size;

    DenseLayer layerQ;
    DenseLayer layerK;
    DenseLayer layerV;
    DenseLayer layerFinal;

    Matrix Q;
    Matrix K;
    Matrix V;
    Matrix attentionMatrix;

    static Matrix remade_matrix_split_head(Matrix const &m, Index context, Index nb_head);
    static Matrix merge_head(Matrix const &m, Index context, Index nb_head);
    static Matrix softmax_masked(Matrix const &score, Index context);

};

#endif //RESEAUDENEURONEFROMSCRATCH_ATTENTIONLAYER_HPP
