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
    Shape s(1,64,64);
    cout << "Chargement du dataset train" << endl;
    DatasetImage dataset_train(s,"chest_xray/train", 0, 77);
    cout << "Chargement terminé" << endl;
    Random gen(87);
    Network net;
    CCE fn_loss;

    net.add(make_unique<ConvLayer>(s,8,3,gen,1));
    net.add(make_unique<ReLU>(30752));
    net.add(make_unique<MaxPoolingLayer>(Shape(8,62,62),2));
    net.add(make_unique<ConvLayer>(Shape(8,31,31),16,3,gen,1));
    net.add(make_unique<ReLU>(13456));
    net.add(make_unique<MaxPoolingLayer>(Shape(16,29,29),2));
    net.add(make_unique<FlattenLayer>(Shape(16,14,14)));
    net.add(make_unique<DenseLayer>(3136,64,gen));
    net.add(make_unique<ReLU>(64));
    net.add(make_unique<DenseLayer>(64,2,gen));
    net.add(make_unique<SoftmaxLayer>(2));

    load(net,"models/chest_xray.txt");

    auto parameters = net.get_parameters();
    Adam optimizer(parameters, 0.001);

    int const EPOCH = 67;
    Timer t;

    cout << "Début de l'entrainement" << endl;
    t.reset();
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        dataset_train.shuffle();
        Index begin = 0;
        Scalar error = 0;
        while (begin < dataset_train.get_nImages()) {
            pair<Matrix, Matrix> batch = dataset_train.get_batch(begin, 32);
            net.set_gradients_zero();
            Matrix pred = net.forward(batch.first);
            Scalar loss = fn_loss.calculate_loss(pred, batch.second);
            net.backward(fn_loss.calculate_gradients_loss(pred, batch.second));
            optimizer.update_parameters(parameters);
            error += loss * static_cast<Scalar>(batch.first.get_row());
            begin += 32;
        }
        error = error / static_cast<Scalar>(dataset_train.get_nImages());
        cout <<"Erreur moyenne à l'epoch "<< epoch <<": "<< error << endl;
        if (epoch%5 == 0) {
            save(net, "models/chest_xray.txt");
            cout << "Sauvagarde du modèle" << endl;
        }

    }
    cout << "Fin de l'entrainement (Temps écoulé en release pour 20 epochs : " << t.elapsed() << "s)" << endl;
    save(net, "models/chest_xray.txt");

    return 0;
}