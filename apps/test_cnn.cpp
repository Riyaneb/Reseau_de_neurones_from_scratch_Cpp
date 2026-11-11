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
    cout << "Chargement du dataset test" << endl;
    DatasetImage dataset_test(s,"chest_xray/test", 0, 77);
    cout << "Chargement terminé" << endl;
    Random gen(87);
    Network net;
    CCE fn_loss;

    Index good_result = 0;

    load(net,"models/chest_xray.txt");
    Index begin = 0;
    cout << "Test en cours..." << endl;
    while (begin < dataset_test.get_nImages()) {
        pair<Matrix, Matrix> batch = dataset_test.get_batch(begin, 64);
        Matrix pred = net.forward(batch.first);
        for (Index i = 0; i < batch.first.get_row(); i++) {
            Index predicted_class = pred.max_index_value_row(i);
            if (batch.second.get_value(i,predicted_class) == 1) {
                good_result++;
            }
        }
        begin += 64;
    }
    cout << "Test terminé. Le réseau CNN à une précision de : " << fixed << setprecision(2) << (static_cast<Scalar>(good_result) / static_cast<Scalar>(dataset_test.get_nImages())) * 100.00 << " %" << endl;

    return 0;
}