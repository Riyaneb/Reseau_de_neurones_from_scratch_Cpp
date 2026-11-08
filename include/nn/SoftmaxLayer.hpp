#ifndef RESEAUDENEURONEFROMSCRATCH_SOFTMAXLAYER_HPP
#define RESEAUDENEURONEFROMSCRATCH_SOFTMAXLAYER_HPP

#include "math/Matrix.hpp"
#include "nn/Layer.hpp"

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(Index nb) : nbInput(nb), input(1,nb), output(1,1) {}
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    Index nInput() const override;
    Index nOutput() const override;
    std::unique_ptr<Layer> clone() const override;
    std::string layer_name() const override {
        return "Softmax";
    }
protected:
    static Scalar fn_softmax(Matrix const &M,Index row,Index column);
    Index nbInput;
    Matrix input;
    Matrix output;
};

#endif //RESEAUDENEURONEFROMSCRATCH_SOFTMAXLAYER_HPP
