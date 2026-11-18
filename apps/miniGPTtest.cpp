#include <iostream>

#include "io/Serializer.hpp"
#include "math/Matrix.hpp"
#include "nn/Network.hpp"
#include "nn/Losses.hpp"
#include "nn/Optimizers.hpp"
#include "llm/GPT.hpp"
#include "utils/DatasetText.hpp"
#include "utils/Timer.hpp"

using namespace std;

int main() {
    cout << "Chargement du dataset test" << endl;
    DatasetText dataset_train("MaxLSB_LeCarnet_dataset/train", 0, 77);
    DatasetText dataset_val("MaxLSB_LeCarnet_dataset/val", 0, 77);
    cout << "Chargement terminé" << endl;
    Random gen;
    Network net = GPT(256,512,8,8,256, gen);
    CCE fn_loss;
    net.to_gpu();

    std::ofstream log_file("data/training_gpt_log.csv");
    if (log_file) {
        log_file << "tour,perte_train,perte_val\n";
    }

    auto parameters = net.get_parameters();
    Adam optimizer(parameters, 3e-4, 0.9, 0.99);

    int const EPOCH = 120000;
    Scalar rMax = 3e-4;
    Scalar rMin = 3e-5;
    Scalar pi = std::acos(-1.0);
    Scalar best_loss = 1e9;
    Scalar loss_mean = 0;
    Index cpt = 0;
    Timer t;
    cout << "Début de l'entrainement" << endl;
    t.reset();
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        pair<Matrix, Matrix> batch = dataset_train.get_batch(16, 256,true);
        net.set_gradients_zero();

        Matrix pred = net.forward(batch.first);

        Scalar loss = fn_loss.calculate_loss(pred, batch.second);
        loss_mean += loss;
        cpt++;
        auto grad_loss = fn_loss.calculate_gradients_loss(pred, batch.second);

        net.backward(grad_loss);
        Scalar r;
        if (epoch < 2000) {
            r = rMax * static_cast<double>(epoch+1) / 2000;
        }
        else {
            Scalar p = static_cast<double>(epoch - 2000) / (EPOCH - 2000);
            r = rMin + 0.5*(rMax - rMin) * (1 + std::cos(pi*p));
        }
        optimizer.set_rate(r);
        optimizer.update_parameters(parameters);

        if (epoch % 50 == 0) {
            cout <<"Erreur à l'epoch "<< epoch <<": "<< loss  << endl;
        }
        if (epoch % 1000 == 0) {
            Scalar loss_val = 0;
            for (Index i = 0; i < 50; i++) {
                auto batch_val = dataset_val.get_batch_in_order(i,16,256,true);
                loss_val += fn_loss.calculate_loss(net.forward(batch_val.first), batch_val.second);
            }
            log_file << epoch << "," << loss_mean / cpt << "," << loss_val / 50 << endl;
            cout <<"Erreur moyenne de val à l'epoch "<< epoch <<": "<< loss_val / 50  << endl;
            if (best_loss > loss_val && epoch %5000 == 0) {
                best_loss = loss_val;
                cout <<"Meilleur version du modèle sauvegardé" << endl;
                save(net,"models/best_gpt.txt");
            }
            if (epoch %5000 == 0) {
                save(net,"models/secours_gpt.txt");
            }
            loss_mean = 0;
            cpt = 0;
        }
    }

    double epoch_t = t.elapsed();
    save(net,"models/gpt_end_training.txt");
    cout << "Fin de l'entrainement (Temps écoulé en release pour " << EPOCH <<  " epochs : " << epoch_t << "s)" << endl;
    return 0;
}