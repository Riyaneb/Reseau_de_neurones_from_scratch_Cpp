#include "nn/Optimizers.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#ifdef USE_CUDA
#include "gpu/Gpu_Optimizer.hpp"
#endif
#define MIN_SIZE_OMP 32768

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
        first_moments_params.emplace_back(it->first->get_row(),it->first->get_column(), 0, it->first->get_in_gpu());
        second_moments_params.emplace_back(it->first->get_row(),it->first->get_column(), 0,it->first->get_in_gpu());
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
        assert(temp1.get_in_gpu() == parameters[i].first->get_in_gpu() && temp2.get_in_gpu() == parameters[i].first->get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");

        if (parameters[i].first->get_in_gpu()) {
            #ifdef USE_CUDA
                Adam_gpu(parameters[i].first->get_gpu_ptr(), temp1.get_gpu_ptr(), temp2.get_gpu_ptr(), rate, epsilon, parameters[i].first->get_row(), parameters[i].first->get_column());
            #else
                std::cerr << "Erreur : CUDA non disponible" << std::endl;
                exit(EXIT_FAILURE);
            #endif
        }
        else {
            #pragma omp parallel for if(parameters[i].first->get_row() * parameters[i].first->get_column() > MIN_SIZE_OMP) schedule(dynamic, 256)
            for (int k = 0 ; k < parameters[i].first->get_row() ; k++ ) {
                for (int j = 0 ; j < parameters[i].first->get_column() ; j++ ) {
                    parameters[i].first->get_value(k,j) -= (rate * temp1.get_value(k,j)) / (std::sqrt(temp2.get_value(k,j)) + epsilon);
                }
            }
        }
    }

}
