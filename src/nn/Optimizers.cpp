#include "nn/Optimizers.hpp"
#include <cmath>
#include <cassert>

void SGD::update_parameters(PairParameters &parameters) {
    PairParameters::iterator it;
    for (it = parameters.begin(); it != parameters.end(); ++it) {
        assert(it->first->get_row() == it->second->get_row() && it->first->get_column() == it->second->get_column() && "Erreur : Paramètre n'ayant pas la même dimension que leur gradient");
        *(it->first) -= rate * *(it->second);
    }
}

Adam::Adam(PairParameters const &pair_parameters, Scalar a, Scalar b1, Scalar b2, Scalar eps) : rate(a), beta1(b1), beta2(b2), epsilon(eps), cpt(0) {
    PairParameters::const_iterator it;
    for (it = pair_parameters.begin(); it != pair_parameters.end(); ++it) {
        first_moments_params.emplace_back(it->first->get_row(),it->first->get_column());
        second_moments_params.emplace_back(it->first->get_row(),it->first->get_column());
    }
}

void Adam::update_parameters(PairParameters &parameters) {
    int size = parameters.size();
    assert(parameters.size() == first_moments_params.size() && parameters.size() == second_moments_params.size() && "Erreur : les conteneurs n'ont pas la même taille");
    cpt++;
    for (int i = 0; i < size; i++) {
        first_moments_params[i] = first_moments_params[i] * beta1 + *(parameters[i].second) * (1 - beta1);
        second_moments_params[i] = second_moments_params[i] * beta2 + hadamard(*(parameters[i].second),*(parameters[i].second)) * (1 - beta2);

        Matrix temp1 = first_moments_params[i] * (1.0 / (1 - std::pow(beta1,cpt)));
        Matrix temp2 = second_moments_params[i] * (1.0 / (1 - std::pow(beta2,cpt)));

        #pragma omp parallel for collapse(2) schedule(dynamic, 256)
        for (int k = 0 ; k < parameters[i].first->get_row() ; k++ ) {
            for (int j = 0 ; j < parameters[i].first->get_column() ; j++ ) {
                parameters[i].first->get_value(k,j) -= (rate * temp1.get_value(k,j)) / (std::sqrt(temp2.get_value(k,j)) + epsilon);
            }
        }
    }

}
