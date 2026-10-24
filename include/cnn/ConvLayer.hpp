#ifndef RESEAUDENEURONEFROMSCRATCH_CONVLAYER_HPP
#define RESEAUDENEURONEFROMSCRATCH_CONVLAYER_HPP
#include "math/Random.hpp"
#include "nn/Layer.hpp"
#include "nn/Shape.hpp"
#include "nn/Initializers.hpp"

class ConvLayer : public Layer {
public:
    ConvLayer(Shape const &s, Index nKernel, Index dimKer, Random &gen, Index initChoice = 1);
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    Index nInput() const override;
    Index nOutput() const override;
    PairParameters get_parameters() override;
    void set_gradients_zero() override;
    std::unique_ptr<Layer> clone() const override;
private:
    Shape shapeIn;
    Shape shapeOut;
    Index sizeKernel;
    Index nKer;

    Matrix weight;
    Matrix biais;
    Matrix gradWeight;
    Matrix gradBiais;

    Matrix im2col;
};

#endif //RESEAUDENEURONEFROMSCRATCH_CONVLAYER_HPP
