#include "nn/Activations.hpp"
#include <cassert>

#ifdef USE_CUDA
#include "gpu/Gpu_Activation.hpp"
#endif

Scalar ReLU::apply(Scalar value) const {
    if (value < 0) {
        return 0;
    }
    else {
        return value;
    }
}

Scalar ReLU::derivate_fn(Scalar const &value) const {
    if (value <= 0) {
        return 0;
    }
    else {
        return 1;
    }
}

Matrix ReLU::forward(Matrix const &in) {
    assert(in.get_column() == nbInput && "Erreur : Dimension dans le forward activationLayer Incorrecte");
    input = in;
    Matrix result(input.get_row(), input.get_column(),0,in.get_in_gpu());
    if (in.get_in_gpu()) {
        #ifdef USE_CUDA
            forward_activation_gpu(in.get_gpu_ptr(), result.get_gpu_ptr(), 0, input.get_row(), input.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
        for (Index i = 0; i < input.get_row(); i++) {
            for (Index j = 0; j < input.get_column(); j++) {
                Scalar temp = input.get_value(i,j);
                if (temp > 0) {
                    result.get_value(i,j) = temp;
                }
            }
        }
    }
    return result;
}

Matrix ReLU::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == input.get_column() && gradientOut.get_row() == input.get_row() && "Dimension de la matrice des gradients ne correspond pas au nombre de neurones de la couche précédente");
    Matrix result(gradientOut.get_row(), gradientOut.get_column(),0, gradientOut.get_in_gpu());
    if (gradientOut.get_in_gpu()) {
        #ifdef USE_CUDA
            backward_activation_gpu(input.get_gpu_ptr(), gradientOut.get_gpu_ptr(), result.get_gpu_ptr(),0, input.get_row(), input.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
        for (Index i = 0; i < input.get_row(); i++) {
            for (Index j = 0; j < input.get_column(); j++) {
                Scalar temp = input.get_value(i,j);
                if (temp > 0) {
                    result.get_value(i,j) = gradientOut.get_value(i,j);
                }
            }
        }
    }
    return result;
}

Matrix Tanh::forward(Matrix const &in) {
    assert(in.get_column() == nbInput && "Erreur : Dimension dans le forward activationLayer Incorrecte");
    input = in;
    Matrix result(input.get_row(), input.get_column(),0,in.get_in_gpu());
    if (in.get_in_gpu()) {
        #ifdef USE_CUDA
            forward_activation_gpu(in.get_gpu_ptr(), result.get_gpu_ptr(), 1, input.get_row(), input.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
        for (Index i = 0; i < input.get_row(); i++) {
            for (Index j = 0; j < input.get_column(); j++) {
                Scalar temp = input.get_value(i,j);
                result.get_value(i,j) = std::tanhf(temp);
            }
        }
    }
    return result;
}

Matrix Tanh::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == input.get_column() && gradientOut.get_row() == input.get_row() && "Dimension de la matrice des gradients ne correspond pas au nombre de neurones de la couche précédente");
    Matrix result(gradientOut.get_row(), gradientOut.get_column(),0, gradientOut.get_in_gpu());
    if (gradientOut.get_in_gpu()) {
        #ifdef USE_CUDA
            backward_activation_gpu(input.get_gpu_ptr(), gradientOut.get_gpu_ptr(), result.get_gpu_ptr(),1, input.get_row(), input.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
        for (Index i = 0; i < input.get_row(); i++) {
            for (Index j = 0; j < input.get_column(); j++) {
                Scalar temp = std::tanhf(input.get_value(i,j));
                result.get_value(i,j) = gradientOut.get_value(i,j) * (1.0f - temp * temp);
            }
        }
    }
    return result;
}

Matrix Sigmoid::forward(Matrix const &in) {
    assert(in.get_column() == nbInput && "Erreur : Dimension dans le forward activationLayer Incorrecte");
    input = in;
    Matrix result(input.get_row(), input.get_column(),0,in.get_in_gpu());
    if (in.get_in_gpu()) {
        #ifdef USE_CUDA
            forward_activation_gpu(in.get_gpu_ptr(), result.get_gpu_ptr(), 2, input.get_row(), input.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
        for (Index i = 0; i < input.get_row(); i++) {
            for (Index j = 0; j < input.get_column(); j++) {
                Scalar temp = std::expf(-input.get_value(i,j));
                result.get_value(i,j) = 1.0f / (1.0f + temp);
            }
        }
    }
    return result;
}

Matrix Sigmoid::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == input.get_column() && gradientOut.get_row() == input.get_row() && "Dimension de la matrice des gradients ne correspond pas au nombre de neurones de la couche précédente");
    Matrix result(gradientOut.get_row(), gradientOut.get_column(),0, gradientOut.get_in_gpu());
    if (gradientOut.get_in_gpu()) {
        #ifdef USE_CUDA
            backward_activation_gpu(input.get_gpu_ptr(), gradientOut.get_gpu_ptr(), result.get_gpu_ptr(),2, input.get_row(), input.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for if(input.get_row() * input.get_column() > MIN_SIZE_OMP) schedule(dynamic,1)
        for (Index i = 0; i < input.get_row(); i++) {
            for (Index j = 0; j < input.get_column(); j++) {
                Scalar temp = 1.0f / (1.0f + std::expf(-input.get_value(i,j)));
                result.get_value(i,j) = gradientOut.get_value(i,j) * temp * (1.0f - temp);
            }
        }
    }
    return result;
}

Matrix Identity::forward(Matrix const &in) {
    assert(in.get_column() == nbInput && "Erreur : Dimension dans le forward activationLayer Incorrecte");
    input = in;
    Matrix result(input.get_row(), input.get_column(),0,in.get_in_gpu());
    if (in.get_in_gpu()) {
        #ifdef USE_CUDA
            forward_activation_gpu(in.get_gpu_ptr(), result.get_gpu_ptr(), 3, input.get_row(), input.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        result = in;
    }
    return result;
}

Matrix Identity::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == input.get_column() && gradientOut.get_row() == input.get_row() && "Dimension de la matrice des gradients ne correspond pas au nombre de neurones de la couche précédente");
    Matrix result(gradientOut.get_row(), gradientOut.get_column(),0, gradientOut.get_in_gpu());
    if (gradientOut.get_in_gpu()) {
        #ifdef USE_CUDA
            backward_activation_gpu(input.get_gpu_ptr(), gradientOut.get_gpu_ptr(), result.get_gpu_ptr(),3, input.get_row(), input.get_column());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        result = gradientOut;
    }
    return result;
}

std::unique_ptr<Layer> ReLU::clone() const {
    return std::make_unique<ReLU>(*this);
}

Scalar Sigmoid::apply(Scalar value) const {
    return 1.0f/(1.0f + std::exp(-value));
}

Scalar Sigmoid::derivate_fn(Scalar const &value) const {
    Scalar temp = Sigmoid::apply(value);
    return temp * (1.0f - temp);
}

std::unique_ptr<Layer> Sigmoid::clone() const {
    return std::make_unique<Sigmoid>(*this);
}

Scalar Tanh::apply(Scalar value) const {
    return std::tanh(value);
}

Scalar Tanh::derivate_fn(Scalar const &value) const {
    return 1.0f - pow(std::tanh(value),2);
}

std::unique_ptr<Layer> Tanh::clone() const {
    return std::make_unique<Tanh>(*this);
}

Scalar Identity::apply(Scalar value) const {
    return value;
}

Scalar Identity::derivate_fn(Scalar const &value) const {
    if (value <= 0) {
    }
    return 1;
}

std::unique_ptr<Layer> Identity::clone() const {
    return std::make_unique<Identity>(*this);
}

