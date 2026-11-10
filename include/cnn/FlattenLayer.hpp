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
    std::string layer_name() const override {
        return "Flatten";
    }
    std::vector<Index> save_config() const override {
        std::vector<Index> config(0);
        config.push_back(shape.get_channel());
        config.push_back(shape.get_height());
        config.push_back(shape.get_width());
        return config;
    }
private:
    Shape shape;
};

#endif //RESEAUDENEURONEFROMSCRATCH_FLATTENLAYER_HPP
