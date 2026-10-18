#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_MATRIX_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_MATRIX_HPP

#include "math/Types.hpp"
#include <vector>

class Matrix {
public:
    Matrix(Index l,Index c,Scalar init_value=0): row(l), column(c), mat(check_size(l,c), init_value) {}
    Index get_row() const;
    Index get_column() const;
    const Scalar &get_value(Index l, Index c) const;
    Scalar &get_value(Index l, Index c);
    Scalar *get_ptr_matrix();
    Scalar const *get_ptr_matrix() const;

    Matrix& operator+=(Matrix const &m);
    Matrix& operator-=(Matrix const &m);
    Matrix& hadamard(Matrix const &m);
    Matrix& operator*=(Scalar const &a);

    bool isSquare() const;

    Matrix transpose() const;
    Matrix& broadcast(Matrix const &m);
    Matrix sum_row() const;
    static Matrix matIdentity(Index i);

    void render();

    Index max_index_value_row(Index r) const;
    Scalar max_value_row(Index r) const;


private:
    static Index check_size(Index l, Index c);
    Index row;
    Index column;
    std::vector<Scalar> mat;
    Index size_mat(Index l, Index c) const;
};

Matrix operator+(Matrix const &m1, Matrix const &m2);
Matrix operator-(Matrix const &m1, Matrix const &m2);
Matrix hadamard(Matrix const &m1, Matrix const &m2);
Matrix operator*(Matrix const &m1, Scalar a);
Matrix operator*(Scalar a, Matrix const &m1);
Matrix broadcast(Matrix const &m, Matrix const &biais);
Matrix operator*(Matrix const &m1, Matrix const &m2);

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_MATRIX_HPP
