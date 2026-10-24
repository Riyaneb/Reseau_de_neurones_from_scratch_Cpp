#include "math/Matrix.hpp"
#include <cassert>
#include <iostream>
#ifdef USE_CUDA
    #include "gpu/Gpu_calcul_matrix.hpp"
#endif
#define MIN_SIZE 100000

Index Matrix::check_size(Index l, Index c) {
    assert(l > 0 && c >0 && "Le nombre de ligne ou de colonne de la matrice est négatif ou nulle");
    return l*c;
}

Index Matrix::size_mat(Index l, Index c) const{
    return l*column + c;
}

Index Matrix::get_row() const {
    return row;
}

Index Matrix::get_column() const {
    return column;
}
Index Matrix::get_size() const {
    return row*column;
}


const Scalar &Matrix::get_value(Index l, Index c) const {
    assert(l>=0 && l < row && c>=0 && c < column && "Erreur : Index incorrect");
    return mat[size_mat(l, c)];
}

Scalar &Matrix::get_value(Index l, Index c) {
    assert(l>=0 && l < row && c>=0 && c < column && "Erreur : Index incorrect");
    return mat[size_mat(l, c)];
}

Scalar* Matrix::get_ptr_matrix() {
    return mat.data();
}

Scalar const *Matrix::get_ptr_matrix() const {
    return mat.data();
}

Matrix& Matrix::operator+=(Matrix const &m) {
    assert(m.row == row && m.column == column && "Erreur : Les matrices n'ont pas la même dimension");
    for (Index i = 0; i < row*column; ++i) {
        mat[i] += m.mat[i];
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
    for (Index i = 0; i < row*column; ++i) {
        mat[i] -= m.mat[i];
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
    for (Index i = 0; i < row*column; ++i) {
        mat[i] *= m.mat[i];
    }
    return *this;
}

Matrix hadamard(Matrix const &m1, Matrix const &m2) {
    Matrix result = Matrix(m1);
    result.hadamard(m2);
    return result;
}

Matrix& Matrix::operator*=(Scalar const &a) {
    for (Index i = 0; i < row*column; ++i) {
        mat[i] *= a;
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

bool Matrix::isSquare() const {
    return row == column;
}

Matrix Matrix::transpose() const {
    Matrix result(column,row);
    for (Index i = 0; i < row; ++i) {
        for (Index j = 0; j < column; ++j) {
            result.get_value(j,i) = this->get_value(i,j);
        }
    }
    return result;
}

Matrix &Matrix::broadcast(Matrix const &biais) {
    assert(biais.row == 1 && biais.column == column && "Erreur : Dimension du biais incorrect");
    for (Index i = 0; i < row; ++i) {
        for (Index j = 0; j < column; ++j) {
            this->get_value(i,j) += biais.get_value(0,j);
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
    Matrix result(1,column);
    for (Index i = 0; i < row; ++i) {
        for (Index j = 0; j < column; ++j) {
            result.get_value(0,j) += this->get_value(i,j);
        }
    }
    return result;
}

Matrix Matrix::multiplyMatrixCPU(Matrix const &m1, Matrix const &m2) {
    Matrix result(m1.get_row(),m2.get_column(),0);
    Scalar* res = result.get_ptr_matrix();
    const Scalar* p1 = m1.get_ptr_matrix();
    const Scalar* p2 = m2.get_ptr_matrix();

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
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            std::cout << this->get_value(i,j) << " ";
        }
        std::cout << std::endl;
    }
}

Index Matrix::max_index_value_row(Index r) const {
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
