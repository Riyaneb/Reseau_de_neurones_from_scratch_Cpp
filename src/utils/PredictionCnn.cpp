#include "utils/PredictionCnn.hpp"

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

Answer_gui prediction_image(DatasetImage& dataset, Index num_image, Shape const &s, string const &path, Index dataset_type) {
    cout << "Chargement du dataset test" << endl;
    cout << "Chargement terminé" << endl;
    Random gen(87);
    Network net;
    CCE fn_loss;

    Scalar score_max = 0;
    Index predicted_class = 0;
    std::string predicted_name;


    std::vector<Scalar> score;
    std::string answer_name;
    bool answer = false;

    net.add(make_unique<ConvLayer>(s,8,3,gen,1));
    net.add(make_unique<ReLU>(5408));
    net.add(make_unique<MaxPoolingLayer>(Shape(8,26,26),2));
    net.add(make_unique<FlattenLayer>(Shape(8,13,13)));
    net.add(make_unique<DenseLayer>(1352,10,gen));
    net.add(make_unique<SoftmaxLayer>(10));

    load(net,"models/reseau_mnist.txt");

    pair<Matrix, Matrix> batch = dataset.get_batch(num_image, 1);
    Matrix pred = net.forward(batch.first);
    for (Index i = 0; i < batch.first.get_row(); i++) {
        predicted_class = pred.max_index_value_row(i);
        score_max = pred.max_value_row(i);
        predicted_name = dataset.get_class_names()[predicted_class];
        answer_name = dataset.get_class_names()[batch.second.max_index_value_row(i)];
        if (batch.second.get_value(i,predicted_class) == 1) {
            answer = true;
        }
    }

    score.reserve(pred.get_column());
    for (Index i = 0; i < pred.get_column(); i++) {
        score.push_back(pred.get_value(0,i));
    }

    return Answer_gui(dataset.get_class_names(), score, predicted_name , score_max*100.0f,answer, predicted_class,answer_name);
}
