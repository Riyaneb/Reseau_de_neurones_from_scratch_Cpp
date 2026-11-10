#include "nn/Network.hpp"

#include "utils/Timer.hpp"


void Network::add(std::unique_ptr<Layer> layer) {
    assert(layer != nullptr && "Erreur : Le pointeur est nul");
    if (!layers.empty()) {
        assert(layer->nInput() == layers.back()->nOutput() && "Erreur : Le nombre d'entrée de la nouvelle couche ne correspond pas au nombre de sortie de la dernière couche");
    }
    else {
        layer->set_need_grad_input(false);
    }
    layers_name.push_back(layer->layer_name());
    layers.push_back(std::move(layer));

}

Matrix Network::forward(Matrix const &X) {
    assert(!layers.empty() && "Erreur : Le réseau est vide");
    assert(X.get_column() == layers[0]->nInput() && "Erreur : Dimension de l'entrée incorrecte pour ce réseau");
    Matrix result = X;
    std::vector<std::unique_ptr<Layer>>::iterator it;
    for (it = layers.begin(); it != layers.end(); ++it) {
        result = (*it)->forward(result);
    }
    return result;
}

Matrix Network::backward(Matrix const &gradLoss) {
    assert(!layers.empty() && "Erreur : Le réseau est vide");
    assert(layers.back()->nOutput() == gradLoss.get_column() && "Erreur : Dimension de sortie incorrecte pour ce réseau");
    Matrix result = gradLoss;
    std::vector<std::unique_ptr<Layer>>::reverse_iterator it;
    for (it = layers.rbegin(); it != layers.rend(); ++it) {
        result = (*it)->backward(result);
    }
    return result;
}


PairParameters Network::get_parameters() {
    assert(!layers.empty() && "Erreur : Le réseau est vide");
    PairParameters parameters;
    std::vector<std::unique_ptr<Layer>>::iterator it;
    for (it = layers.begin(); it != layers.end(); ++it) {
        PairParameters::iterator it2;
        PairParameters temp = (*it)->get_parameters();
        for (it2 = temp.begin(); it2 != temp.end(); ++it2) {
            parameters.push_back(*it2);
        }
    }
    return parameters;
}

void Network::set_gradients_zero() {
    assert(!layers.empty() && "Erreur : Le réseau est vide");
    std::vector<std::unique_ptr<Layer>>::iterator it;
    for (it = layers.begin(); it != layers.end(); ++it) {
        (*it)->set_gradients_zero();
    }
}

Network Network::clone() const {
    Network net;
    std::vector<std::unique_ptr<Layer>>::const_iterator it;
    for (it = layers.begin(); it != layers.end(); ++it) {
        net.add((*it)->clone());
    }
    return net;
}

void Network::to_cpu() {
    PairParameters::iterator it;
    auto params = get_parameters();
    for (it = params.begin(); it != params.end(); ++it) {
        it->first->to_cpu();
        it->second->to_cpu();
    }
}

void Network::to_gpu() {
    PairParameters::iterator it;
    auto params = get_parameters();
    for (it = params.begin(); it != params.end(); ++it) {
        it->first->to_gpu();
        it->second->to_gpu();
    }
}

bool Network::in_gpu() {
    auto params = get_parameters();
    if (params.empty()) {
        return false;
    }
    return params.begin()->first->get_in_gpu();
}
