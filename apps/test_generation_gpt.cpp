#include <iostream>
#include <ctime>
#include "io/Serializer.hpp"
#include "math/Matrix.hpp"
#include "nn/Network.hpp"
#include "nn/Losses.hpp"
#include "nn/Optimizers.hpp"
#include "llm/GPT.hpp"
#include "utils/DatasetText.hpp"
#include "utils/Timer.hpp"
#include <windows.h>

using namespace std;

int main() {
    SetConsoleOutputCP(CP_UTF8);
    cout << "Chargement du dataset test" << endl;
    DatasetText dataset_test("MaxLSB_LeCarnet_dataset/test", 0, 77);
    cout << "Chargement terminé" << endl;
    Network net;
    load(net,"models/gpt_end_training.txt",true);
    time_t timestamp = time(nullptr);
    Random gen;
    gen.changeSeed(timestamp);
    Index line = gen.randomizeBigInt(0,dataset_test.get_nb_lines()-1);
    auto text_string = dataset_test.get_line(line);

    if (text_string.size() >= 4 && text_string.compare(text_string.size()-4, 4,"***\n") == 0) {
        text_string.erase(text_string.size() - 4, 4);
    }
    else if (text_string.size() >= 3 && text_string.compare(text_string.size()-3, 3,"***") == 0) {
        text_string.erase(text_string.size() - 3, 3);
    }

    auto text_octet = DatasetText::encode(text_string);
    auto temp = text_octet;

    if (text_octet.size() > 256) {
        text_octet.erase(text_octet.begin(), text_octet.end() - 256);
    }

    Index size = text_octet.size();

    cout << "Texte dont on veut générer la suite : " << DatasetText::decode(temp) << "\n" << endl;
    std::uint8_t next_letter;

    Timer t;
    cout << "Début de génération..." << endl;
    t.reset();

    Index b = 0;

    do {
        Matrix X(text_octet.size(),1);
        for (Index i = 0; i < text_octet.size(); ++i) {
            X.get_value(i,0) = text_octet[i];
        }
        X.to_gpu();
        net.set_actual_context(text_octet.size());
        auto Y = net.forward(X);
        Y.to_cpu();

        Scalar sum = 0;
        Scalar heat;
        for (Index j = 0; j < 256; ++j) {
            Scalar calcul = std::pow(Y.get_value(Y.get_row()-1,j),1/0.8);
            Y.get_value(Y.get_row()-1,j) = calcul;
            sum += calcul;
        }

        for (Index j = 0; j < 256; ++j) {
            Y.get_value(Y.get_row()-1,j) /= sum;
        }

        Scalar ref = gen.randomizeUniform(0,1);
        Scalar base = 0;
        next_letter = 0;

        for (Index j = 0; j < 256; ++j) {
            base += Y.get_value(Y.get_row()-1,next_letter);
            if (base >= ref) {
                break;
            }
            next_letter++;
        }


        temp.push_back(next_letter);
        text_octet.push_back(next_letter);
        if (text_octet.size() > 256) {
            text_octet.erase(text_octet.begin(), text_octet.end() - 256);
        }
        b++;

    }while (DatasetText::decode(text_octet).back() != '\n' && b <= 800);


    cout << "Fin de génération : " << endl << endl << DatasetText::decode(temp) << endl << endl << "Temps d'exécution : " << t.elapsed() << "s" << endl;
    std::ofstream file("data/dataset_text/MaxLSB_LeCarnet_dataset/result/result_test_gpt_LeCarnet");
    if (!file) {
        std::cerr << "Erreur : Le fichier n'a pas pu être crée" << endl;
    }
    else {
        file << DatasetText::decode(temp);
    }
    return 0;
}