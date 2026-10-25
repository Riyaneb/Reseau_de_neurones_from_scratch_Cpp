#ifndef RESEAUDENEURONEFROMSCRATCH_MAXPOOLINGLAYER_HPP
#define RESEAUDENEURONEFROMSCRATCH_MAXPOOLINGLAYER_HPP
#include "math/Random.hpp"
#include "nn/Layer.hpp"
#include "nn/Shape.hpp"

class MaxPoolingLayer : public Layer {
public:
    MaxPoolingLayer(Shape const &s, Index dimKer);
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    Index nInput() const override;
    Index nOutput() const override;
    std::unique_ptr<Layer> clone() const override;

    static std::pair<Scalar, Index> max(std::vector<std::pair<Scalar, Index>> const &tab);
private:
    Shape shapeIn;
    Shape shapeOut;
    Index sizeKernel;
    Matrix posMaxim;
};

#endif //RESEAUDENEURONEFROMSCRATCH_MAXPOOLINGLAYER_HPP
