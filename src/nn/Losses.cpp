#include "nn/Losses.hpp"
#include <algorithm>
#ifdef USE_CUDA
#include "gpu/Gpu_Loss.hpp"
#endif
#define MIN_SIZE_OMP 32768

Scalar QuadraLoss::calculate_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    assert(prediction.get_in_gpu() == target.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    Scalar res;
    if (prediction.get_in_gpu()) {
        #ifdef USE_CUDA
            Matrix loss_value(1,1,0,prediction.get_in_gpu());
            calcul_loss_gpu(prediction.get_gpu_ptr(), target.get_gpu_ptr(), loss_value.get_gpu_ptr(), 0, prediction.get_row(), prediction.get_column());
            loss_value.to_cpu();
            res = loss_value.get_value(0,0);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        Scalar loss_value = 0;
        #pragma omp parallel for if(prediction.get_row() * prediction.get_column() > MIN_SIZE_OMP) reduction(+:loss_value) schedule(dynamic,256)
        for (Index i = 0; i < prediction.get_row(); i++) {
            for (Index j = 0; j < prediction.get_column(); j++) {
                loss_value += (target.get_value(i,j) - prediction.get_value(i,j)) * (target.get_value(i,j) - prediction.get_value(i,j));
            }
        }
        loss_value /= 2.0f * static_cast<Scalar>(prediction.get_row());
        res = loss_value;
    }
    return res;
}

Matrix QuadraLoss::calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    assert(prediction.get_in_gpu() == target.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    Matrix grad = prediction - target;
    if (prediction.get_in_gpu()) {
        #ifdef USE_CUDA
            gradient_loss_gpu(prediction.get_gpu_ptr(), target.get_gpu_ptr(), grad.get_gpu_ptr(), 0, prediction.get_row(), prediction.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        Scalar n = static_cast<Scalar>(prediction.get_row());
        #pragma omp parallel for if(grad.get_row() * grad.get_column() > MIN_SIZE_OMP) schedule(dynamic,256)
        for (Index i = 0; i < grad.get_row(); i++) {
            for (Index j = 0; j < grad.get_column(); j++) {
                grad.get_value(i,j) = grad.get_value(i,j) / n;
            }
        }
    }
    return grad;
}


Scalar BCE::calculate_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    assert(prediction.get_in_gpu() == target.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    Scalar res;
    if (prediction.get_in_gpu()) {
        #ifdef USE_CUDA
            Matrix loss_value(1,1,0,prediction.get_in_gpu());
            calcul_loss_gpu(prediction.get_gpu_ptr(), target.get_gpu_ptr(), loss_value.get_gpu_ptr(), 1, prediction.get_row(), prediction.get_column());
            loss_value.to_cpu();
            res = loss_value.get_value(0,0);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        Scalar loss_value = 0;
        #pragma omp parallel for if(prediction.get_row() * prediction.get_column() > MIN_SIZE_OMP) reduction(+:loss_value) schedule(dynamic,256)
        for (int i = 0; i < prediction.get_row(); i++) {
            for (int j = 0; j < prediction.get_column(); j++) {
                Scalar y = target.get_value(i,j);
                Scalar p = std::clamp(prediction.get_value(i,j),1e-7f,1 - 1e-7f);
                loss_value += y * std::logf(p) + (1-y) * std::logf(1-p);
            }
        }
        loss_value = (-1) * (loss_value / (static_cast<Scalar>(prediction.get_row())));
        res = loss_value;
    }
    return res;
}

Matrix BCE::calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    assert(prediction.get_in_gpu() == target.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    Index n = prediction.get_row();
    Index c = prediction.get_column();
    Matrix grad(n,c,0,prediction.get_in_gpu());

    if (prediction.get_in_gpu()) {
        #ifdef USE_CUDA
            gradient_loss_gpu(prediction.get_gpu_ptr(), target.get_gpu_ptr(), grad.get_gpu_ptr(), 1, n, c);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(grad.get_row() * grad.get_column() > MIN_SIZE_OMP) schedule(dynamic,256)
        for (Index i = 0; i < grad.get_row(); i++) {
            for (Index j = 0; j < grad.get_column(); j++) {
                Scalar y = target.get_value(i,j);
                Scalar p = std::clamp(prediction.get_value(i,j),1e-7f,1 - 1e-7f);
                grad.get_value(i,j) = (p-y)/(p*(1-p));
                grad.get_value(i,j) = grad.get_value(i,j) / n;
            }
        }
    }
    return grad;
}

Scalar CCE::calculate_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    assert(prediction.get_in_gpu() == target.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    Scalar res;

    if (prediction.get_in_gpu()) {
        #ifdef USE_CUDA
            Matrix loss_value(1,1,0,prediction.get_in_gpu());
            calcul_loss_gpu(prediction.get_gpu_ptr(), target.get_gpu_ptr(), loss_value.get_gpu_ptr(), 2, prediction.get_row(), prediction.get_column());
            loss_value.to_cpu();
            res = loss_value.get_value(0,0);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        Scalar loss_value = 0;
        #pragma omp parallel for if(prediction.get_row() * prediction.get_column() > MIN_SIZE_OMP) reduction(+:loss_value) schedule(dynamic,256)
        for (int i = 0; i < prediction.get_row(); i++) {
            for (int j = 0; j < prediction.get_column(); j++) {
                Scalar y = target.get_value(i,j);
                Scalar p = prediction.get_value(i,j);
                if (p < 1e-7) {
                    p = 1e-7;
                }
                loss_value += y * std::logf(p);
            }
        }
        loss_value = (-1) * (loss_value / (static_cast<Scalar>(prediction.get_row())));
        res = loss_value;
    }
    return res;
}

Matrix CCE::calculate_gradients_loss(Matrix const &prediction, Matrix const &target) const {
    assert(prediction.get_row() == target.get_row() && prediction.get_column() == target.get_column() && "Erreur : Les matrices n'ont pas la même dimension");
    assert(prediction.get_in_gpu() == target.get_in_gpu() && "Erreur : Les matrices ne sont pas sur le même device");
    Index n = prediction.get_row();
    Index c = prediction.get_column();
    Matrix grad(n,c, 0, prediction.get_in_gpu());

    if (prediction.get_in_gpu()) {
        #ifdef USE_CUDA
            gradient_loss_gpu(prediction.get_gpu_ptr(), target.get_gpu_ptr(), grad.get_gpu_ptr(), 2, n, c);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(grad.get_row() * grad.get_column() > MIN_SIZE_OMP) schedule(dynamic,256)
        for (Index i = 0; i < grad.get_row(); i++) {
            for (Index j = 0; j < grad.get_column(); j++) {
                Scalar y = target.get_value(i,j);
                Scalar p = prediction.get_value(i,j);
                if (p < 1e-7) {
                    p = 1e-7;
                }
                grad.get_value(i,j) = (-1) * (y/p);
                grad.get_value(i,j) = grad.get_value(i,j) / n;
            }
        }
    }
    return grad;
}