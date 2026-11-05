#include <sstream>
#include "math/Matrix.hpp"
#include "TestRunner.hpp"
#include "nn/Initializers.hpp"

void comparaison_matrix(TestRunner &runner, Matrix const &result, Matrix const &expected, std::string const &nom);

void test_matrix_to_gpu(TestRunner &runner) {
    Matrix A(1,1,7);
    Matrix B(1,7);
    Matrix C(64,784);
    Matrix D(43264,9);

    Random gen;
    init_weight_Glorot(B,gen);
    init_weight_Glorot(C,gen);
    init_weight_Glorot(D,gen);

    Matrix c_A = A;
    Matrix c_B = B;
    Matrix c_C = C;
    Matrix c_D = D;

    runner.section("Test envoie vers gpu");

    A.to_gpu();
    B.to_gpu();
    C.to_gpu();
    D.to_gpu();

    runner.check_values(true, A.get_in_gpu(), "A est sur GPU");
    runner.check_values(true, B.get_in_gpu(), "B est sur GPU");
    runner.check_values(true, C.get_in_gpu(), "C est sur GPU");
    runner.check_values(true, D.get_in_gpu(), "D est sur GPU");

    A.to_cpu();
    B.to_cpu();
    C.to_cpu();
    D.to_cpu();

    runner.section("Test envoie vers cpu");

    runner.check_values(false, A.get_in_gpu(), "A est sur CPU");
    runner.check_values(false, B.get_in_gpu(), "B est sur CPU");
    runner.check_values(false, C.get_in_gpu(), "C est sur CPU");
    runner.check_values(false, D.get_in_gpu(), "D est sur CPU");

    runner.section("Test intégrité des matrices");

    comparaison_matrix_big(runner,A,c_A,"Verification de l'intégrité de A");
    comparaison_matrix_big(runner,B,c_B,"Vérification de l'intégrité de B");
    comparaison_matrix_big(runner, C,c_C, "Verification de l'intégrité de C");
    comparaison_matrix_big(runner, D, c_D, "Verification de l'intégrité de D");

}

void test_copy_matrix(TestRunner &runner) {
    Matrix A(3,4);
    Random gen;
    init_weight_Glorot(A,gen);
    Matrix c_A = A;

    runner.section("Test copie sur une nouvelle matrice");
    A.to_gpu();
    Matrix B = A;
    B.to_cpu();
    comparaison_matrix(runner, B,c_A,"Vérification B=A avec B nouveau");


    runner.section("Indépendance des matrices");
    B.get_value(0,0) = 7;
    B.to_gpu();
    A.to_cpu();
    comparaison_matrix(runner, A,c_A,"Indépendance de A et de B");


    runner.section("Copie d'une matrice dans une autre déjà existante");
    Matrix C(1,1);
    C.to_gpu();
    A.to_gpu();
    C = A;
    C.to_cpu();
    comparaison_matrix(runner, C,c_A,"Vérification C=A avec C déjà existante");

    runner.section("Copie d'une matrice dans elle meme");
    A = A;
    A.to_cpu();
    comparaison_matrix(runner, A, c_A, "A=A");


    runner.section("Vérification d'une matrice sur gpu déplacement dans une matrice nouvelle");
    A.to_gpu();
    Scalar* ptr_A = A.get_gpu_ptr();
    Matrix E = std::move(A);
    Scalar* ptr_E = E.get_gpu_ptr();

    runner.check_values(ptr_A, ptr_E, "Vérification de la récuperation de E du pointeur de A");
    bool answer = A.get_gpu_ptr() == nullptr;
    runner.check_values(true,answer,"Vérificationd e la dispariton de A dans le gpu");
    E.to_cpu();
    comparaison_matrix(runner, E, c_A, "E doit etre identique à A");

    runner.section("Vérification d'une matrice sur gpu déplacement dans une matrice déjà existante");
    C.to_gpu();
    E.to_gpu();
    ptr_E = E.get_gpu_ptr();
    C = std::move(E);
    Scalar* ptr_C = C.get_gpu_ptr();
    runner.check_values(ptr_E, ptr_C, "Vérification de la récuperation de C du pointeur de E");
    C.to_cpu();
    comparaison_matrix(runner, C, c_A, "C doit etre identique à E");

}

void test_general_matrix_gpu(TestRunner &runner) {
    test_matrix_to_gpu(runner);
    test_copy_matrix(runner);
}