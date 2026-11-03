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

    auto names_layers = net.get_names();

    auto parameters = net.get_parameters();
    Adam optimizer(parameters, 0.001);

    int const EPOCH = 1;
    Timer t;
    Timer t2;

    double batch_t = 0, grad_zero_t = 0, loss_t = 0, opti_t = 0;

    std::vector<double> f_t(names_layers.size(), 0.0);
    std::vector<double> b_t(names_layers.size(), 0.0);

    cout << "Début de l'entrainement" << endl;
    t.reset();
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        dataset_train.shuffle();
        Index begin = 0;
        Scalar error = 0;
        while (begin < dataset_train.get_nImages()) {
            t2.reset();
            pair<Matrix, Matrix> batch = dataset_train.get_batch(begin, 64);
            batch_t += t2.elapsed();

            t2.reset();
            net.set_gradients_zero();
            grad_zero_t += t2.elapsed();


            Matrix pred = net.forward(batch.first);
            auto forward_times = net.get_forward_times();
            for (Index i = 0; i < names_layers.size(); i++) {
                f_t[i] += forward_times[i];
            }

            t2.reset();
            Scalar loss = fn_loss.calculate_loss(pred, batch.second);
            auto grad_loss = fn_loss.calculate_gradients_loss(pred, batch.second);
            loss_t += t2.elapsed();

            net.backward(grad_loss);
            auto backward_times = net.get_backward_times();
            for (Index i = 0; i < names_layers.size(); i++) {
                b_t[i] += backward_times[names_layers.size() - i - 1];
            }

            t2.reset();
            optimizer.update_parameters(parameters);
            opti_t += t2.elapsed();

            error += loss * static_cast<Scalar>(batch.first.get_row());
            begin += 64;
        }
        error = error / static_cast<Scalar>(dataset_train.get_nImages());
        cout <<"Erreur moyenne à l'epoch "<< epoch <<": "<< error << endl;

    }

    double epoch_t = t.elapsed();
    cout << "Fin de l'entrainement (Temps écoulé en release pour 1 epochs : " << epoch_t << "s)" << endl;
    double sum_t = batch_t + grad_zero_t + loss_t + opti_t;

    cout << "\nTemps par couche :" << endl;
    for (Index i = 0; i < names_layers.size(); i++) {
        cout << "Couche " << i << " " << names_layers[i] << " : forward " << f_t[i] << "s, backward " << b_t[i] << "s" << endl;
        sum_t += f_t[i] + b_t[i];
    }

    cout << "\nAutres étapes :" << endl;
    cout << "Création du batch : " << batch_t << "s" << endl;
    cout << "Remise à zero des gradients : " << grad_zero_t << "s" << endl;
    cout << fn_loss.loss_name() << " : " << loss_t << "s" << endl;
    cout << optimizer.optimiser_name() << " : " << opti_t << "s" << endl;

    cout << "\nTemps non mesuré : " << epoch_t - sum_t << "s" << endl;

    cout << "\nDétail backward Convolution :" << endl;
    cout << "Rangement gradRes : " << ConvLayer::b_gradRes_t << "s" << endl;
    cout << "gradWeight : " << ConvLayer::b_gradWeight_t << "s" << endl;
    cout << "gradBiais : " << ConvLayer::b_gradBiais_t << "s" << endl;
    cout << "Reste : " << b_t[0] - ConvLayer::b_gradRes_t - ConvLayer::b_gradWeight_t - ConvLayer::b_gradBiais_t << "s" << endl;


    cout << "\nDétail forward Convolution :" << endl;
    cout << "im2col : " << ConvLayer::f_im2col_t << "s" << endl;
    cout << "Produit : " << ConvLayer::f_line_t << "s" << endl;
    cout << "broadcast : " << ConvLayer::f_broadcast_t << "s" << endl;
    cout << "Rangement sortie : " << ConvLayer::f_res_t << "s" << endl;
    cout << "Reste : " << f_t[0] - ConvLayer::f_im2col_t - ConvLayer::f_line_t - ConvLayer::f_broadcast_t - ConvLayer::f_res_t << "s" << endl;

    save(net, "models/reseau_mnist.txt");

    return 0;
}