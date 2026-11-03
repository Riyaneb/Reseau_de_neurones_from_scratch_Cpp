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

    runner.check_values(false, A.get_in_gpu(), "A est sur CPU");
    runner.check_values(false, B.get_in_gpu(), "B est sur CPU");
    runner.check_values(false, C.get_in_gpu(), "C est sur CPU");
    runner.check_values(false, D.get_in_gpu(), "D est sur CPU");

    comparaison_matrix_big(runner,A,c_A,"Verification de l'intégrité de A");
    comparaison_matrix_big(runner,B,c_B,"Vérification de l'intégrité de B");
    comparaison_matrix_big(runner, C,c_C, "Verification de l'intégrité de C");
    comparaison_matrix_big(runner, D, c_D, "Verification de l'intégrité de D");

}

void test_general_matrix_gpu(TestRunner &runner) {
    test_matrix_to_gpu(runner);
}