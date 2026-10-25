#ifndef RESEAUDENEURONEFROMSCRATCH_FLATTENLAYER_HPP
#define RESEAUDENEURONEFROMSCRATCH_FLATTENLAYER_HPP
#include "nn/Layer.hpp"
#include "nn/Shape.hpp"

class FlattenLayer : public Layer {
public:
    FlattenLayer(Shape s) : shape(s) {}
    Matrix forward(Matrix const &in) override;
    Matrix backward(Matrix const &gradientOut) override;
    Index nInput() const override;
    Index nOutput() const override;
    std::unique_ptr<Layer> clone() const override;

private:
    Shape shape;
};

#endif //RESEAUDENEURONEFROMSCRATCH_FLATTENLAYER_HPP
