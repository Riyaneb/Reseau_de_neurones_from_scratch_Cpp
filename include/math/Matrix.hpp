#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_MATRIX_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_MATRIX_HPP

#include <cassert>

#include "math/Types.hpp"
#include <vector>
#include <utility>
#define MIN_SIZE_OMP 32768


class Matrix {
public:
    Matrix(Index l,Index c,Scalar init_value=0): row(l), column(c), mat(check_size(l,c), init_value), in_gpu(false), gpu_ptr(nullptr) {}
    Matrix(Matrix const &m);
    Matrix(Matrix &&m) noexcept;
    ~Matrix();

    inline Index get_row() const {
        return row;
    }

    inline Index get_column() const {
        return column;
    }

    Index get_size() const {
        return row*column;
    }

    const Scalar &get_value(Index l, Index c) const {
        assert(!in_gpu && "Erreur : La matrice n'est pas sur CPU");
        assert(l>=0 && l < row && c>=0 && c < column && "Erreur : Index incorrect");
        return mat[size_mat(l, c)];
    }

    Scalar &get_value(Index l, Index c) {
        assert(!in_gpu && "Erreur : La matrice n'est pas sur CPU");
        assert(l>=0 && l < row && c>=0 && c < column && "Erreur : Index incorrect");
        return mat[size_mat(l, c)];
    }

    bool get_in_gpu() const {
        return in_gpu;
    }

    Scalar* get_gpu_ptr() {
        return gpu_ptr;
    }

    const Scalar* get_gpu_ptr() const {
        return gpu_ptr;
    }

    void to_gpu();
    void to_cpu();

    Scalar *get_ptr_matrix();
    Scalar const *get_ptr_matrix() const;

    Matrix& operator+=(Matrix const &m);
    Matrix& operator-=(Matrix const &m);
    Matrix& hadamard(Matrix const &m);
    Matrix& operator*=(Scalar const &a);
    Matrix& operator=(Matrix const &m);
    Matrix& operator=(Matrix &&m) noexcept;

    static Matrix multiplyMatrixCPU(Matrix const &m1, Matrix const &m2);
    static Matrix multiplyMatrixGPU(Matrix const &m1, Matrix const &m2);

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
    bool in_gpu;
    Scalar* gpu_ptr;
    inline Index size_mat(Index l, Index c) const {
        return l * column + c;
    }
};

Matrix operator+(Matrix const &m1, Matrix const &m2);
Matrix operator-(Matrix const &m1, Matrix const &m2);
Matrix hadamard(Matrix const &m1, Matrix const &m2);
Matrix operator*(Matrix const &m1, Scalar a);
Matrix operator*(Scalar a, Matrix const &m1);
Matrix broadcast(Matrix const &m, Matrix const &biais);
Matrix operator*(Matrix const &m1, Matrix const &m2);

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_MATRIX_HPP
