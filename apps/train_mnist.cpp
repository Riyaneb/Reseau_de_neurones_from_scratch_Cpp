#include <iostream>
#include "math/Matrix.hpp"
#include "nn/Network.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Activations.hpp"
#include "nn/Losses.hpp"
#include "nn/Optimizers.hpp"
#include "nn/SoftmaxLayer.hpp"
#include "cnn/ConvLayer.hpp"
#include "cnn/FlattenLayer.hpp"
#include "cnn/MaxPoolingLayer.hpp"
#include <sstream>

#include "io/Serializer.hpp"
#include "utils/DatasetImage.hpp"
#include "utils/Timer.hpp"

using namespace std;


int main() {
    Shape s(1,28,28);
    cout << "Chargement du dataset train" << endl;
    DatasetImage dataset_train(s,"MNIST/train", 1, 77);
    cout << "Chargement terminé" << endl;
    Random gen(87);
    Network net;
    CCE fn_loss;

    net.add(make_unique<ConvLayer>(s,8,3,gen,1));
    net.add(make_unique<ReLU>(5408));
    net.add(make_unique<MaxPoolingLayer>(Shape(8,26,26),2));
    net.add(make_unique<FlattenLayer>(Shape(8,13,13)));
    net.add(make_unique<DenseLayer>(1352,10,gen));
    net.add(make_unique<SoftmaxLayer>(10));

    auto parameters = net.get_parameters();
    Adam optimizer(parameters, 0.001);

    int const EPOCH = 10;
    Timer t;

    cout << "Début de l'entrainement" << endl;
    t.reset();
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        dataset_train.shuffle();
        Index begin = 0;
        Scalar error = 0;
        while (begin < dataset_train.get_nImages()) {
            pair<Matrix, Matrix> batch = dataset_train.get_batch(begin, 64);
            net.set_gradients_zero();
            Matrix pred = net.forward(batch.first);
            Scalar loss = fn_loss.calculate_loss(pred, batch.second);
            net.backward(fn_loss.calculate_gradients_loss(pred, batch.second));
            optimizer.update_parameters(parameters);
            error += loss * static_cast<Scalar>(batch.first.get_row());
            begin += 64;
        }
        error = error / static_cast<Scalar>(dataset_train.get_nImages());
        cout <<"Erreur moyenne à l'epoch "<< epoch <<": "<< error << endl;

    }
    cout << "Fin de l'entrainement (Temps écoulé en release pour 10 epochs : " << t.elapsed() << "s)" << endl;
    save(net, "models/reseau_mnist.txt");

    return 0;
}