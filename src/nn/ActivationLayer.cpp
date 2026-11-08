#include "nn/ActivationLayer.hpp"
#include <cassert>
#define MIN_SIZE_OMP 32768

Index ActivationLayer::nInput() const {
    return nbInput;
}

Index ActivationLayer::nOutput() const {
    return nbInput;
}
