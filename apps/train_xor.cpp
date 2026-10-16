#include <iostream>
#include "math/Matrix.hpp"
#include "nn/Network.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Activations.hpp"
#include "nn/Losses.hpp"
#include "nn/Optimizers.hpp"
#include "nn/SoftmaxLayer.hpp"
#include <sstream>

#include "io/Serializer.hpp"
#include "utils/Logger.hpp"


int main() {

    Matrix input_xor(4, 2);
    input_xor.get_value(0, 0) = 0.0;
    input_xor.get_value(0, 1) = 0.0;

    input_xor.get_value(1, 0) = 0.0;
    input_xor.get_value(1, 1) = 1.0;

    input_xor.get_value(2, 0) = 1.0;
    input_xor.get_value(2, 1) = 0.0;

    input_xor.get_value(3, 0) = 1.0;
    input_xor.get_value(3, 1) = 1.0;

    Matrix target_xor(4, 2);

    target_xor.get_value(0, 0) = 1.0;
    target_xor.get_value(0, 1) = 0.0;

    target_xor.get_value(1, 0) = 0.0;
    target_xor.get_value(1, 1) = 1.0;

    target_xor.get_value(2, 0) = 0.0;
    target_xor.get_value(2, 1) = 1.0;

    target_xor.get_value(3, 0) = 1.0;
    target_xor.get_value(3, 1) = 0.0;

    Random gen(33);
    BCE fn_loss;
    // SGD optimizer(taux);

    Network net;

    auto ptrDense1 = std::make_unique<DenseLayer>(2, 4, gen);
    net.add(std::move(ptrDense1));
    net.add(std::make_unique<Tanh>(4));

    auto ptrDense2 = std::make_unique<DenseLayer>(4, 2, gen);
    net.add(std::move(ptrDense2));
    net.add(std::make_unique<Sigmoide>(2));

    const Index EPOCH = 5000;

    PairParameters parameters = net.get_parameters();
    Adam optimizer2(parameters,0.01);
    for (Index i = 0; i < EPOCH; i++) {
        net.set_gradients_zero();
        Matrix pred = net.forward(input_xor);
        Scalar error = fn_loss.calculate_loss(pred, target_xor);
        if (i%250 == 0) {
            std::cout <<"Erreur à l'epoch "<< i <<": "<< error << std::endl;
        }
        net.backward(fn_loss.calculate_gradients_loss(pred, target_xor));
        optimizer2.update_parameters(parameters);
    }

    Matrix pred = net.forward(input_xor);
    comparaison_matrix(pred, target_xor, "Test reseau xor");

    std::cout <<"Perte finale : "<< fn_loss.calculate_loss(pred,target_xor) << std::endl;

    save(net, "models/reseau_xor.txt");
}
