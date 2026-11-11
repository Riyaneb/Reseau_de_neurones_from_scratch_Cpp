#include <cuda_runtime_api.h>
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
    Network netCPU;
    CCE fn_loss;

    netCPU.add(make_unique<ConvLayer>(s,8,3,gen,1));
    netCPU.add(make_unique<ReLU>(5408));
    netCPU.add(make_unique<MaxPoolingLayer>(Shape(8,26,26),2));
    netCPU.add(make_unique<FlattenLayer>(Shape(8,13,13)));
    netCPU.add(make_unique<DenseLayer>(1352,10,gen));
    netCPU.add(make_unique<SoftmaxLayer>(10));

    Network netGPU = netCPU.clone();
    netGPU.to_gpu();

    auto parametersCPU = netCPU.get_parameters();
    auto parametersGPU = netGPU.get_parameters();
    Adam optimizerCPU(parametersCPU, 0.001);
    Adam optimizerGPU(parametersGPU, 0.001);


    Timer tCPU;
    Timer tGPU;

    double timeCPU = 0, timeGPU = 0;
    cout << "Début de l'entrainement" << endl;

    const Index EPOCH= 10;
    for (Index epoch = 0; epoch < EPOCH; epoch++) {
        dataset_train.shuffle();
        Index begin = 0;
        Scalar errorCPU = 0;
        Scalar errorGPU = 0;
        while (begin < dataset_train.get_nImages()) {
            pair<Matrix, Matrix> batchCPU = dataset_train.get_batch(begin, 64);
            tCPU.reset();
            netCPU.set_gradients_zero();
            Matrix predCPU = netCPU.forward(batchCPU.first);
            Scalar lossCPU = fn_loss.calculate_loss(predCPU, batchCPU.second);
            auto grad_lossCPU = fn_loss.calculate_gradients_loss(predCPU, batchCPU.second);
            netCPU.backward(grad_lossCPU);
            optimizerCPU.update_parameters(parametersCPU);
            errorCPU += lossCPU * static_cast<Scalar>(batchCPU.first.get_row());
            timeCPU += tCPU.elapsed();

            auto batchGPU = batchCPU;
            batchGPU.first.to_gpu();
            batchGPU.second.to_gpu();
            tGPU.reset();
            netGPU.set_gradients_zero();
            Matrix predGPU = netGPU.forward(batchGPU.first);
            Scalar lossGPU = fn_loss.calculate_loss(predGPU, batchGPU.second);
            auto grad_lossGPU = fn_loss.calculate_gradients_loss(predGPU, batchGPU.second);
            netGPU.backward(grad_lossGPU);
            optimizerGPU.update_parameters(parametersGPU);
            errorGPU += lossGPU * static_cast<Scalar>(batchGPU.first.get_row());
            cudaDeviceSynchronize();
            timeGPU += tGPU.elapsed();
            begin += 64;
        }
    }


    cout << "Temps moyen par epoch sur CPU : " << timeCPU / EPOCH << endl;
    cout << "Temps moyen par epoch sur GPU : " << timeGPU / EPOCH << endl;

    DatasetImage dataset_test(s,"MNIST/test", 1, 77);
    Index good_resultCPU = 0;
    Index begin = 0;
    Index good_resultGPU = 0;
    cout << "Test en cours..." << endl;
    while (begin < dataset_test.get_nImages()) {
        pair<Matrix, Matrix> batchCPU = dataset_test.get_batch(begin, 64);
        auto batchGPU = batchCPU;
        batchGPU.first.to_gpu();

        Matrix predCPU = netCPU.forward(batchCPU.first);
        Matrix predGPU = netGPU.forward(batchGPU.first);

        predGPU.to_cpu();

        for (Index i = 0; i < batchCPU.first.get_row(); i++) {
            Index predicted_class = predCPU.max_index_value_row(i);
            if (batchCPU.second.get_value(i,predicted_class) == 1) {
                good_resultCPU++;
            }
        }
        for (Index i = 0; i < batchGPU.first.get_row(); i++) {
            Index predicted_class = predGPU.max_index_value_row(i);
            if (batchCPU.second.get_value(i,predicted_class) == 1) {
                good_resultGPU++;
            }
        }
        begin += 64;
    }

    cout << "Test terminé. Le réseau CNN à une précision sur CPU de : " << fixed << setprecision(2) << (static_cast<Scalar>(good_resultCPU) / static_cast<Scalar>(dataset_test.get_nImages())) * 100.00 << " %" << endl;
    cout << "Test terminé. Le réseau CNN à une précision sur GPU de : " << fixed << setprecision(2) << (static_cast<Scalar>(good_resultGPU) / static_cast<Scalar>(dataset_test.get_nImages())) * 100.00 << " %" << endl;

    return 0;
}