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
    std::string layer_name() const override {
        return "MaxPooling";
    }
    std::vector<Index> save_config() const override {
        std::vector<Index> config;
        config.push_back(shapeIn.get_channel());
        config.push_back(shapeIn.get_height());
        config.push_back(shapeIn.get_width());
        config.push_back(sizeKernel);
        return config;
    }
private:
    Shape shapeIn;
    Shape shapeOut;
    Index sizeKernel;
    Matrix posMaxim;
};

#endif //RESEAUDENEURONEFROMSCRATCH_MAXPOOLINGLAYER_HPP
