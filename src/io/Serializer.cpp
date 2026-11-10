#include "io/Serializer.hpp"

#include "cnn/ConvLayer.hpp"
#include "cnn/FlattenLayer.hpp"
#include "cnn/MaxPoolingLayer.hpp"
#include "nn/Activations.hpp"
#include "nn/SoftmaxLayer.hpp"

bool save(Network &net, std::string const &filename) {
    std::ofstream f(filename.c_str());
    bool in_gpu = net.in_gpu();
    if (f) {
        if (in_gpu) {
            net.to_cpu();
        }
        f << std::setprecision(std::numeric_limits<Scalar>::max_digits10);

        auto& layers = net.get_layers();
        Index size = layers.size();
        f << size << std::endl;

        for ( auto it = layers.begin() ; it != layers.end() ; ++it) {
            std::string name = (*it)->layer_name();

            if (name != "DenseLayer" && name != "Convolution" && name != "MaxPooling" && name != "Flatten" && name != "ReLU" && name != "Sigmoid" && name != "Tanh" && name != "Identity" && name != "Softmax") {
                std::cerr << "Erreur dans la sauvegarde du réseau : " << name << "n'est pas défini dans le réseau" << std::endl;
                exit(EXIT_FAILURE);
            }

            f << name;
            auto config = (*it)->save_config();
            for (auto it2 = config.begin(); it2 != config.end(); ++it2) {
                f << " " << (*it2);
            }
            f << std::endl;

            if (name == "DenseLayer" || name == "Convolution") {
                auto param = (*it)->get_parameters();
                Index nRowWeight = param[0].first->get_row();
                Index nColWeight = param[0].first->get_column();

                Index nRowBiais = param[1].first->get_row();
                Index nColBiais = param[1].first->get_column();

                f << nRowWeight << " " << nColWeight << " " << nRowBiais << " " << nColBiais << std::endl;

                for (Index i = 0; i < nRowWeight; i++) {
                    for (Index j = 0; j < nColWeight; j++) {
                        f << param[0].first->get_value(i,j) << " ";
                    }
                }

                for (Index i = 0; i < nRowBiais; i++) {
                    for (Index j = 0; j < nColBiais; j++) {
                        f << param[1].first->get_value(i,j) << " ";
                    }
                }
            }
            f << std::endl;
        }
    }
    else {
        std::cerr << "Erreur dans la création du fichier " << filename << "\n";
        return false;
    }
    f.close();
    if (in_gpu) {
        net.to_gpu();
    }
    return true;
}

bool load(Network &net, std::string const &filename, bool in_gpu) {
    std::ifstream f(filename.c_str());
    assert(net.get_layers().empty() && "Erreur : Le réseau doit être vide");
    if (f) {
        Index fileSize;
        f >> fileSize;
        for (Index k = 0; k < fileSize; k++) {
            std::string name;

            f >> name;

            if (name == "DenseLayer") {
                Random gen;
                Index rowD, colD;
                f >> rowD >> colD;
                DenseLayer layer(rowD, colD, gen);

                Index nRowWeight;
                Index nColWeight;
                Index nRowBiais;
                Index nColBiais;

                f >> nRowWeight >> nColWeight >> nRowBiais >> nColBiais;
                auto params = layer.get_parameters();

                if (nRowWeight != params[0].first->get_row() || nColWeight != params[0].first->get_column() || nRowBiais != params[1].first->get_row() || nColBiais != params[1].first->get_column()) {
                    std::cerr << "Erreur : Les dimensions des paramètres de la couche " << name << " (Couche " << k << ") et ceux du fichier ne correspondent pas" << std::endl;
                    return false;
                }

                for (Index i = 0; i < nRowWeight; i++) {
                    for (Index j = 0; j < nColWeight; j++) {
                        f >> params[0].first->get_value(i,j);
                    }
                }

                for (Index i = 0; i < nRowBiais; i++) {
                    for (Index j = 0; j < nColBiais; j++) {
                        f >> params[1].first->get_value(i,j);
                    }
                }

                net.add(std::make_unique<DenseLayer>(layer));

            }
            else if (name == "Convolution") {
                Random gen;
                Index c,h,w,nKer,dimKer;
                f >> c >> h >> w >> nKer >> dimKer;
                ConvLayer layer(Shape(c,h,w), nKer, dimKer, gen);

                Index nRowWeight;
                Index nColWeight;
                Index nRowBiais;
                Index nColBiais;

                f >> nRowWeight >> nColWeight >> nRowBiais >> nColBiais;
                auto params = layer.get_parameters();

                if (nRowWeight != params[0].first->get_row() || nColWeight != params[0].first->get_column() || nRowBiais != params[1].first->get_row() || nColBiais != params[1].first->get_column()) {
                    std::cerr << "Erreur : Les dimensions des paramètres de la couche " << name << " (Couche " << k << ") et ceux du fichier ne correspondent pas" << std::endl;
                    return false;
                }

                for (Index i = 0; i < nRowWeight; i++) {
                    for (Index j = 0; j < nColWeight; j++) {
                        f >> params[0].first->get_value(i,j);
                    }
                }

                for (Index i = 0; i < nRowBiais; i++) {
                    for (Index j = 0; j < nColBiais; j++) {
                        f >> params[1].first->get_value(i,j);
                    }
                }

                net.add(std::make_unique<ConvLayer>(layer));
            }
            else if (name == "MaxPooling") {
                Index c,h,w,dimKer;
                f >> c >> h >> w >> dimKer;
                net.add(std::make_unique<MaxPoolingLayer>(Shape(c,h,w), dimKer));
            }
            else if (name == "Flatten") {
                Index c,h,w;
                f >> c >> h >> w;
                net.add(std::make_unique<FlattenLayer>(Shape(c,h,w)));
            }
            else if (name == "ReLU") {
                Index rowA;
                f >> rowA;
                net.add(std::make_unique<ReLU>(rowA));
            }
            else if (name == "Sigmoid") {
                Index rowA;
                f >> rowA;
                net.add(std::make_unique<Sigmoid>(rowA));
            }
            else if (name == "Tanh") {
                Index rowA;
                f >> rowA;
                net.add(std::make_unique<Tanh>(rowA));
            }
            else if (name == "Identity") {
                Index rowA;
                f >> rowA;
                net.add(std::make_unique<Identity>(rowA));
            }
            else if (name == "Softmax") {
                Index rowA;
                f >> rowA;
                net.add(std::make_unique<SoftmaxLayer>(rowA));
            }
            else {
                std::cerr << "Erreur dans le chargement du réseau : " << name << "n'est pas défini dans le réseau" << std::endl;
                return false;
            }
        }
        if (in_gpu) {
            net.to_gpu();
        }
    }
    else {
        std::cerr << "Erreur dans l'ouverture du fichier " << filename << "\n";
        return false;
    }
    f.close();
    return true;
}
