#include "TestRunner.hpp"
#include "math/Matrix.hpp"
#include "math/Random.hpp"
#include <sstream>

void test_indexation(TestRunner &runner, Matrix &matrix);
void test_matrice(TestRunner& runner);

void test_random(TestRunner &runner);
void test_initializers(TestRunner &runner);

void test_layer(TestRunner& runner);

void test_loss(TestRunner &runner);
void test_backward(TestRunner &runner);
void test_gradient_check_network(TestRunner &runner);
void test_gradient_check_complet(TestRunner &runner);

void test_clone(TestRunner& runner);

void test_corridor(TestRunner &runner);

void test_replay_buffer(TestRunner &runner);

void test_max_value_row(TestRunner &runner);

void test_snake_env(TestRunner &runner);

void test_general_matrix_gpu(TestRunner &runner);

int main() {
    TestRunner runner;
    test_matrice(runner);
    test_random(runner);
    test_initializers(runner);
    test_layer(runner);
    test_loss(runner);
    test_backward(runner);
    test_gradient_check_network(runner);
    test_gradient_check_complet(runner);
    test_clone(runner);
    test_corridor(runner);
    test_replay_buffer(runner);
    test_max_value_row(runner);
    test_snake_env(runner);
    test_general_matrix_gpu(runner);

    std::cout << std::endl;
    runner.bilan();
    return runner.return_test();
}