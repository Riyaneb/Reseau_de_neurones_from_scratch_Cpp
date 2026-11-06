#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>
#include <omp.h>
#include <mimalloc-new-delete.h>

#ifdef USE_CUDA
    #include "gpu/Gpu_calcul_matrix.hpp"
    #include "gpu/Memory_manage_matrix.hpp"
#endif
#define MIN_SIZE 100000000

Matrix::Matrix(Matrix const &m) : row(m.row), column(m.column), mat(m.mat), in_gpu(m.in_gpu), gpu_ptr(nullptr) {
    if (in_gpu) {
        #ifdef USE_CUDA
            size_t octets = static_cast<size_t>(row)*column*sizeof(Scalar);
            gpu_ptr = gpu_alloc(get_size());
            CUDA_CHECK(cudaMemcpy(gpu_ptr, m.gpu_ptr, octets, cudaMemcpyDeviceToDevice));
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
}

Matrix::Matrix(Matrix &&m) noexcept : row(m.row), column(m.column), mat(std::move(m.mat)), in_gpu(m.in_gpu), gpu_ptr(m.get_gpu_ptr()) {
    m.gpu_ptr = nullptr;
    m.in_gpu = false;
}

Index Matrix::check_size(Index l, Index c) {
    assert(l > 0 && c >0 && "Le nombre de ligne ou de colonne de la matrice est négatif ou nulle");
    return l*c;
}
Scalar* Matrix::get_ptr_matrix() {
    assert(!in_gpu && "Erreur : La matrice n'est pas sur CPU");
    return mat.data();
}

Scalar const *Matrix::get_ptr_matrix() const {
    assert(!in_gpu && "Erreur : La matrice n'est pas sur CPU");
    return mat.data();
}

Matrix& Matrix::operator+=(Matrix const &m) {
    assert(m.row == row && m.column == column && "Erreur : Les matrices n'ont pas la même dimension");
    assert(in_gpu == m.in_gpu && "Erreur : Les matrices ne sont pas sur le même device");
    if (in_gpu) {
        #ifdef USE_CUDA
        plusEqual_gpu(gpu_ptr, m.gpu_ptr, column, row);
        #endif
    }
    else {
        #pragma omp parallel for if(column * row > MIN_SIZE_OMP) schedule(dynamic,8192)
        for (Index i = 0; i < row*column; ++i) {
            mat[i] += m.mat[i];
        }
    }
    return *this;
}

Matrix operator+(Matrix const &m1, Matrix const &m2) {
    Matrix result = Matrix(m1);
    result += m2;
    return result;
}

Matrix& Matrix::operator-=(Matrix const &m) {
    assert(m.row == row && m.column == column && "Erreur : Les matrices n'ont pas la même dimension");
    assert(in_gpu == m.in_gpu && "Erreur : Les matrices ne sont pas sur le même device");
    if (in_gpu) {
        #ifdef USE_CUDA
        minusEqual_gpu(gpu_ptr, m.gpu_ptr, column, row);
        #endif
    }
    else {
        #pragma omp parallel for if(column * row > MIN_SIZE_OMP) schedule(dynamic,8192)
        for (Index i = 0; i < row*column; ++i) {
            mat[i] -= m.mat[i];
        }
    }
    return *this;
}

Matrix operator-(Matrix const &m1, Matrix const &m2) {
    Matrix result = Matrix(m1);
    result -= m2;
    return result;
}

Matrix& Matrix::hadamard(Matrix const &m) {
    assert(m.row == row && m.column == column && "Erreur : Les matrices n'ont pas la même dimension");
    assert(in_gpu == m.in_gpu && "Erreur : Les matrices ne sont pas sur le même device");
    if (in_gpu) {
        #ifdef USE_CUDA
        hadamard_gpu(gpu_ptr, m.gpu_ptr, column, row);
        #endif
    }
    else {
        #pragma omp parallel for if(column * row > MIN_SIZE_OMP) schedule(dynamic,8192)
        for (Index i = 0; i < row*column; ++i) {
            mat[i] *= m.mat[i];
        }
    }
    return *this;
}

Matrix hadamard(Matrix const &m1, Matrix const &m2) {
    Matrix result = Matrix(m1);
    result.hadamard(m2);
    return result;
}

Matrix& Matrix::operator*=(Scalar const &a) {
    if (in_gpu) {
        #ifdef USE_CUDA
        multScalar_gpu(gpu_ptr, a, column, row);
        #endif
    }
    else {
        #pragma omp parallel for if(column * row > MIN_SIZE_OMP) schedule(dynamic,8192)
        for (Index i = 0; i < row*column; ++i) {
            mat[i] *= a;
        }
    }
    return *this;
}

Matrix operator*(Matrix const &m, Scalar a) {
    Matrix result = Matrix(m);
    result *= a;
    return result;
}

Matrix operator*(Scalar a,Matrix const &m) {
    Matrix result = Matrix(m);
    result *= a;
    return result;
}

Matrix &Matrix::operator=(Matrix const &m) {
    if (this == &m) {
        return *this;
    }
    row = m.row;
    column = m.column;
    if (!m.in_gpu) {
        #ifdef USE_CUDA
            if (gpu_ptr != nullptr) {
                gpu_free(gpu_ptr);
                gpu_ptr = nullptr;
            }
        #endif
        mat = m.mat;
        in_gpu = false;
    }
    else {
        #ifdef USE_CUDA
            mat.clear();
            mat.shrink_to_fit();
            size_t octets = static_cast<size_t>(row)*column*sizeof(Scalar);
            gpu_free(gpu_ptr);
            gpu_ptr = gpu_alloc(get_size());
            CUDA_CHECK(cudaMemcpy(gpu_ptr, m.gpu_ptr, octets, cudaMemcpyDeviceToDevice));
            in_gpu = true;
        #else
                std::cerr << "Erreur : CUDA non disponible" << std::endl;
                exit(EXIT_FAILURE);
        #endif
    }
    return *this;
}

Matrix &Matrix::operator=(Matrix &&m) noexcept {
    if (this == &m) {
        return *this;
    }
    row = m.row;
    column = m.column;
    if (!m.in_gpu) {
        #ifdef USE_CUDA
        if (gpu_ptr != nullptr) {
            gpu_free(gpu_ptr);
            gpu_ptr = nullptr;
        }
        #endif
        mat = std::move(m.mat);
        in_gpu = false;
    }
    else {
    #ifdef USE_CUDA
        mat.clear();
        mat.shrink_to_fit();
        gpu_free(gpu_ptr);
        gpu_ptr = m.get_gpu_ptr();
        m.gpu_ptr = nullptr;
        in_gpu = true;
        m.in_gpu = false;
    #else
        std::cerr << "Erreur : CUDA non disponible" << std::endl;
        exit(EXIT_FAILURE);
    #endif
    }
    return *this;
}

bool Matrix::isSquare() const {
    return row == column;
}

Matrix Matrix::transpose() const {
    Matrix result(column,row);
    if (in_gpu) {
        #ifdef USE_CUDA
        result.to_gpu();
        transpose_gpu(result.gpu_ptr, gpu_ptr, column, row);
        #endif
    }
    else {
        #pragma omp parallel for if(row*column > MIN_SIZE_OMP) schedule(dynamic,2048)
        for (Index i = 0; i < row; ++i) {
            for (Index j = 0; j < column; ++j) {
                result.get_value(j,i) = this->get_value(i,j);
            }
        }
    }
    return result;
}

Matrix &Matrix::broadcast(Matrix const &biais) {
    assert(biais.row == 1 && biais.column == column && "Erreur : Dimension du biais incorrect");
    assert(in_gpu == m.in_gpu && "Erreur : Les matrices ne sont pas sur le même device");
    if (in_gpu) {
        #ifdef USE_CUDA
        broadcast_gpu(gpu_ptr, biais.gpu_ptr, column, row);
        #endif
    }
    else {
        #pragma omp parallel for if(row*column > MIN_SIZE_OMP) schedule(dynamic,2048)
        for (Index i = 0; i < row; ++i) {
            for (Index j = 0; j < column; ++j) {
                this->get_value(i,j) += biais.get_value(0,j);
            }
        }
    }
    return *this;
}

Matrix broadcast(Matrix const &m, Matrix const &biais) {
    Matrix result = Matrix(m);
    result.broadcast(biais);
    return result;
}

Matrix Matrix::sum_row() const {
    Matrix result(1,column,0);
    if (in_gpu) {
        #ifdef USE_CUDA
        result.to_gpu();
        sum_row_gpu(result.gpu_ptr, gpu_ptr, column, row);
        #endif
    }
    else {
        long long chunk = 16384 / (column);
        if (chunk < 1) {
            chunk = 1;
        }
            #pragma omp parallel if(row*column > MIN_SIZE_OMP)
        {
            Matrix temp(1,column,0);

            #pragma omp for schedule(dynamic,chunk)
            for (Index i = 0; i < row; ++i) {
                for (Index j = 0; j < column; ++j) {
                    temp.get_value(0,j) += this->get_value(i,j);
                }
            }

            #pragma omp critical
            {
                for (Index i = 0; i < column; ++i) {
                    result.get_value(0,i) += temp.get_value(0,i);
                }
            }
        }
    }
    return result;
}

Matrix Matrix::multiplyMatrixCPU(Matrix const &m1, Matrix const &m2) {
    Matrix result(m1.get_row(),m2.get_column(),0);
    Scalar* res = result.get_ptr_matrix();
    const Scalar* p1 = m1.get_ptr_matrix();
    const Scalar* p2 = m2.get_ptr_matrix();

    long long chunk = 16384 / (m1.get_column() * m2.get_column());
    long long chunk2 = 16384 / (m1.get_row() * m2.get_column());

    if (chunk < 1) {
        chunk = 1;
    }

    if (chunk2 < 1) {
        chunk2 = 1;
    }

    if (m1.get_row() < omp_get_max_threads() && m1.get_column() >= 1024) {
#pragma omp parallel
        {
            std::vector<Scalar> temp(m1.get_row() * m2.get_column(), 0);

#pragma omp for schedule(dynamic,chunk2)
            for (Index k = 0; k < m1.get_column(); ++k){
                for (Index i = 0; i < m1.get_row(); ++i) {
                    Scalar val_m1 = p1[i * m1.get_column() + k];
                    for (Index j = 0; j < m2.get_column(); ++j) {
                        temp[i*m2.get_column() + j] += val_m1 * p2[k * m2.get_column() + j];
                    }
                }
            }
#pragma omp critical
            {
                for (Index i = 0; i < m1.get_row() * m2.get_column(); ++i) {
                    res[i] += temp[i];
                }
            }
        }
    }
    else {
        #pragma omp parallel for if(static_cast<long long>(m1.get_row() * m1.get_column() * m2.get_column()) > MIN_SIZE_OMP) schedule(dynamic,chunk)
        for (Index i = 0; i < m1.get_row(); ++i) {
            Scalar* res_i = res + i * m2.get_column();
            for (Index k = 0; k < m1.get_column(); ++k) {
                Scalar val_m1 = p1[i * m1.get_column() + k];
                const Scalar* m2_k = p2 + k * m2.get_column();
                for (Index j = 0; j < m2.get_column(); ++j) {
                    res_i[j] += val_m1 * m2_k[j];
                }
            }
        }
    }
    return result;
}

Matrix Matrix::multiplyMatrixGPU(Matrix const &m1, Matrix const &m2) {
    assert(m1.get_column() == m2.get_row() && "Erreur : Dimension incorrecte pour la multiplication matricielle");
    #ifdef USE_CUDA
        Index c(m2.get_column()), r(m1.get_row());
        if (gpu_available()) {
            Matrix result(r,c,0);
            gpu_matrix_multiply(m1,m2,result);
            return result;
        }
        else {
            std::cerr << "Aucun GPU utilisant CUDA trouvé" << std::endl;
            exit(EXIT_FAILURE);
        }
    #else
        exit(EXIT_FAILURE);
    #endif
}

Matrix operator*(Matrix const &m1, Matrix const &m2) {
    assert(m1.get_column() == m2.get_row() && "Erreur : Dimension incorrecte pour la multiplication matricielle");
    #ifdef USE_CUDA
        Index c(m2.get_column()), r(m1.get_row());
        long long size = (long long)r*c*m1.get_column();
        if (gpu_available() && size > MIN_SIZE) {
            return Matrix::multiplyMatrixGPU(m1,m2);
        }
    #endif
    return Matrix::multiplyMatrixCPU(m1,m2);
}




Matrix Matrix::matIdentity(Index n) {
    Matrix result(n,n,0);
    for (Index j=0;j<n;j++) {
        result.get_value(j,j) = 1.0;
    }
    return result;
}

void Matrix::render() {
    assert(!in_gpu && "Erreur : La matrice n'est pas sur CPU");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            std::cout << this->get_value(i,j) << " ";
        }
        std::cout << std::endl;
    }
}

Index Matrix::max_index_value_row(Index r) const {
    assert(!in_gpu && "Erreur : La matrice n'est pas sur CPU");
    assert(r >= 0 && r < row && "Erreur : La ligne est invalide");
    Index max_index = 0;
    Scalar max_value = this->get_value(r,0);
    for (Index i = 0; i < column; ++i) {
        Scalar temp = this->get_value(r,i);
        if (temp > max_value) {
            max_value = temp;
            max_index = i;
        }
    }
    return max_index;
}

Scalar Matrix::max_value_row(Index r) const {
    return this->get_value(r,max_index_value_row(r));
}


void Matrix::to_gpu() {
    #ifdef USE_CUDA
    if (!in_gpu) {
        const size_t octets = static_cast<size_t>(row) * column * sizeof(Scalar);
        gpu_ptr = gpu_alloc(get_size());
        CUDA_CHECK(cudaMemcpy(gpu_ptr,mat.data(),octets,cudaMemcpyHostToDevice));
        mat.clear();
        mat.shrink_to_fit();
        in_gpu = true;
    }
    #else
    std::cerr << "Erreur : CUDA non disponible" << std::endl;
    exit(EXIT_FAILURE);
    #endif
}

void Matrix::to_cpu() {
    #ifdef USE_CUDA
    if (in_gpu) {
        const size_t octets = static_cast<size_t>(row) * column * sizeof(Scalar);
        mat.resize(row*column,0);
        CUDA_CHECK(cudaMemcpy(mat.data(), gpu_ptr,octets,cudaMemcpyDeviceToHost));
        gpu_free(gpu_ptr);
        gpu_ptr = nullptr;
        in_gpu = false;
    }
    #else
    std::cerr << "Erreur : CUDA non disponible" << std::endl;
    exit(EXIT_FAILURE);
    #endif
}

Matrix::~Matrix() {
#ifdef USE_CUDA
    if (gpu_ptr != nullptr) {
        gpu_free(gpu_ptr);
    }
#endif
}