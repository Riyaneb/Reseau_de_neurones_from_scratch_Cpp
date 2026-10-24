#include "TestRunner.hpp"
#include "math/Matrix.hpp"
#include "math/Random.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Activations.hpp"
#include "nn/Network.hpp"
#include "nn/Losses.hpp"

void comparaison_matrix(TestRunner &runner, Matrix const &result, Matrix const &expected, std::string const &nom);

void test_loss(TestRunner &runner) {
    QuadraLoss loss_fn;

    Matrix predictions(2, 1);
    predictions.get_value(0, 0) = 3;
    predictions.get_value(1, 0) = 5;

    Matrix targets(2, 1);
    targets.get_value(0, 0) = 1;
    targets.get_value(1, 0) = 2;

    runner.section("Test QuadraLoss : Valeur calculee");

    Scalar expected_loss = 3.25;
    Scalar result_loss = loss_fn.calculate_loss(predictions, targets);

    runner.check_values(expected_loss, result_loss, "Valeur de la perte quadratique");

    runner.section("Test QuadraLoss : Gradient calcule");

    Matrix expected_gradient(2, 1);
    expected_gradient.get_value(0, 0) = 1.0;
    expected_gradient.get_value(1, 0) = 1.5;

    Matrix result_gradient = loss_fn.calculate_gradients_loss(predictions, targets);

    comparaison_matrix(runner, result_gradient, expected_gradient, "Gradient de la perte quadratique");


    runner.section("Test QuadraLoss : predictions identiques aux cibles");

    Scalar perfect_loss = loss_fn.calculate_loss(targets, targets);
    runner.check_values(0.0, perfect_loss, "Perte parfaite vaut 0");

    Matrix expected_perfect_gradient(2, 1);
    expected_perfect_gradient.get_value(0, 0) = 0;
    expected_perfect_gradient.get_value(1, 0) = 0;

    Matrix perfect_gradient = loss_fn.calculate_gradients_loss(targets, targets);

    comparaison_matrix(runner, perfect_gradient, expected_perfect_gradient, "Gradient parfait");
}

void test_backward(TestRunner &runner) {
    Random gen(42);

    runner.section("Test dimension backward");

    DenseLayer dense(3, 2, gen);

    Matrix input_dense(4, 3);
    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 3; ++j) {
            input_dense.get_value(i, j) = 1.0;
        }
    }

    Matrix grad_out_dense(4, 2);
    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 2; ++j) {
            grad_out_dense.get_value(i, j) = 0.5;
        }
    }

    dense.forward(input_dense);
    Matrix grad_in_dense = dense.backward(grad_out_dense);

    runner.check_values(3, dense.get_gradWeight().get_row(), "Lignes gradWeight");
    runner.check_values(2, dense.get_gradWeight().get_column(), "Colonnes gradWeight");

    runner.check_values(1, dense.get_gradBiais().get_row(), "Lignes gradBiais");
    runner.check_values(2, dense.get_gradBiais().get_column(), "Colonnes gradBiais");

    runner.check_values(4, grad_in_dense.get_row(), "Lignes gradient renvoye");
    runner.check_values(3, grad_in_dense.get_column(), "Colonnes gradient renvoye");

    runner.section("Test Tanh Backward ");

    Tanh tanh_layer(3);

    Matrix input_tanh(4, 3);
    Matrix grad_out_tanh(4, 3);
    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 3; ++j) {
            input_tanh.get_value(i, j) = 0.0;
            grad_out_tanh.get_value(i, j) = 1.0;
        }
    }

    tanh_layer.forward(input_tanh);
    Matrix grad_in_tanh = tanh_layer.backward(grad_out_tanh);

    runner.check_values(4, grad_in_tanh.get_row(), "Lignes gradient renvoye (Tanh)");
    runner.check_values(3, grad_in_tanh.get_column(), "Colonnes gradient renvoye (Tanh)");

    runner.section("Test Identity Backward");

    Identity id_layer(3);
    Matrix input_id(2, 3);
    Matrix grad_out_id(2, 3);

    for (Index i = 0; i < 2; ++i) {
        for (Index j = 0; j < 3; ++j) {
            input_id.get_value(i, j) = i * 10 + j;
            grad_out_id.get_value(i, j) = i + j + 1;
        }
    }

    id_layer.forward(input_id);
    Matrix grad_in_id = id_layer.backward(grad_out_id);

    comparaison_matrix(runner, grad_in_id, grad_out_id, "Gradient Identity == Gradient recu");

    runner.section("Test ReLU : Backward (Valeurs et conditions)");

    ReLU relu_layer(3);
    Matrix input_relu(2, 3);

    input_relu.get_value(0, 0) = 5.0;
    input_relu.get_value(0, 1) = -2.0;
    input_relu.get_value(0, 2) = 0.0;
    input_relu.get_value(1, 0) = -10.0;
    input_relu.get_value(1, 1) = 3.14;
    input_relu.get_value(1, 2) = -0.5;

    Matrix grad_out_relu(2, 3);
    for (Index i = 0; i < 2; ++i) {
        for (Index j = 0; j < 3; ++j) {
            grad_out_relu.get_value(i, j) = 10.0;
        }
    }

    Matrix expected_grad_relu(2, 3);
    expected_grad_relu.get_value(0, 0) = 10.0;
    expected_grad_relu.get_value(0, 1) = 0.0;
    expected_grad_relu.get_value(0, 2) = 0.0;
    expected_grad_relu.get_value(1, 0) = 0.0;
    expected_grad_relu.get_value(1, 1) = 10.0;
    expected_grad_relu.get_value(1, 2) = 0.0;

    relu_layer.forward(input_relu);
    Matrix grad_in_relu = relu_layer.backward(grad_out_relu);

    comparaison_matrix(runner, grad_in_relu, expected_grad_relu, "Application du masque de gradient par ReLU");
}

bool check_gradient_relative_error(Scalar grad_analytique, Scalar grad_numerique) {
    Scalar diff = std::abs(grad_analytique - grad_numerique);
    Scalar denom = std::max(1e-8f, std::abs(grad_analytique) + std::abs(grad_numerique));
    Scalar erreur_relative = diff / denom;

    return erreur_relative < 1e-7;
}


void test_gradient_check_network(TestRunner &runner) {
    runner.section("Test Gradient Check");

    Random gen(42);
    Network net;
    QuadraLoss loss_fn;

    auto ptr_layer = std::make_unique<DenseLayer>(2, 3, gen);
    DenseLayer* dense_ptr = ptr_layer.get();
    net.add(std::move(ptr_layer));

    Matrix X(1, 2);
    X.get_value(0, 0) = 0.5;
    X.get_value(0, 1) = -0.2;

    Matrix Y(1, 3);
    Y.get_value(0, 0) = 0.1;
    Y.get_value(0, 1) = 0.8;
    Y.get_value(0, 2) = -0.3;

    Matrix pred_ana = net.forward(X);
    Matrix grad_loss = loss_fn.calculate_gradients_loss(pred_ana, Y);
    net.backward(grad_loss);

    Scalar grad_ana_w = dense_ptr->get_gradWeight().get_value(0, 0);
    Scalar grad_ana_b = dense_ptr->get_gradBiais().get_value(0, 0);

    Scalar epsilon = 1e-5;


    Matrix w = dense_ptr->get_weight();
    Scalar orig_w = w.get_value(0, 0);


    w.get_value(0, 0) = orig_w + epsilon;
    dense_ptr->set_weight(w);
    Scalar loss_plus_w = loss_fn.calculate_loss(net.forward(X), Y);


    w.get_value(0, 0) = orig_w - epsilon;
    dense_ptr->set_weight(w);
    Scalar loss_minus_w = loss_fn.calculate_loss(net.forward(X), Y);


    w.get_value(0, 0) = orig_w;
    dense_ptr->set_weight(w);

    Scalar grad_num_w = (loss_plus_w - loss_minus_w) / (2.0 * epsilon);

    Matrix b = dense_ptr->get_biais();
    Scalar orig_b = b.get_value(0, 0);


    b.get_value(0, 0) = orig_b + epsilon;
    dense_ptr->set_biais(b);
    Scalar loss_plus_b = loss_fn.calculate_loss(net.forward(X), Y);


    b.get_value(0, 0) = orig_b - epsilon;
    dense_ptr->set_biais(b);
    Scalar loss_minus_b = loss_fn.calculate_loss(net.forward(X), Y);

    b.get_value(0, 0) = orig_b;
    dense_ptr->set_biais(b);

    Scalar grad_num_b = (loss_plus_b - loss_minus_b) / (2.0 * epsilon);

    runner.check_values(true, check_gradient_relative_error(grad_ana_w, grad_num_w), "GradCheck Reseau : Poids(0,0)");
    runner.check_values(true, check_gradient_relative_error(grad_ana_b, grad_num_b), "GradCheck Reseau : Biais(0,0)");
}

void test_gradient_check_complet(TestRunner &runner) {
    Random gen(42);
    QuadraLoss loss_fn;
    Scalar epsilon = 1e-5;

    runner.section("Verification sur une couche dense seule");

    DenseLayer denseSingle(3, 2, gen);
    Matrix inputSingle(2, 3);
    inputSingle.get_value(0, 0) = 0.5; inputSingle.get_value(0, 1) = -0.2; inputSingle.get_value(0, 2) = 0.1;
    inputSingle.get_value(1, 0) = -0.4; inputSingle.get_value(1, 1) = 0.8; inputSingle.get_value(1, 2) = 0.3;

    Matrix targetSingle(2, 2);
    targetSingle.get_value(0, 0) = 0.1; targetSingle.get_value(0, 1) = 0.9;
    targetSingle.get_value(1, 0) = -0.3; targetSingle.get_value(1, 1) = 0.4;

    Matrix predSingle = denseSingle.forward(inputSingle);
    Matrix gradLossSingle = loss_fn.calculate_gradients_loss(predSingle, targetSingle);
    Matrix gradInputAna = denseSingle.backward(gradLossSingle);

    Matrix gradWeightAna = denseSingle.get_gradWeight();
    Matrix gradBiaisAna = denseSingle.get_gradBiais();

    bool weightSingleOk = true;
    Matrix w = denseSingle.get_weight();
    for (Index i = 0; i < w.get_row(); ++i) {
        for (Index j = 0; j < w.get_column(); ++j) {
            Scalar orig = w.get_value(i, j);

            w.get_value(i, j) = orig + epsilon; denseSingle.set_weight(w);
            Scalar loss_plus = loss_fn.calculate_loss(denseSingle.forward(inputSingle), targetSingle);

            w.get_value(i, j) = orig - epsilon; denseSingle.set_weight(w);
            Scalar loss_minus = loss_fn.calculate_loss(denseSingle.forward(inputSingle), targetSingle);

            w.get_value(i, j) = orig; denseSingle.set_weight(w);

            Scalar grad_num = (loss_plus - loss_minus) / (2.0 * epsilon);
            if (!check_gradient_relative_error(gradWeightAna.get_value(i, j), grad_num)) {
                weightSingleOk = false;
            }
        }
    }
    runner.check_values(true, weightSingleOk, "Gradients Poids");

    bool biaisSingleOk = true;
    Matrix b = denseSingle.get_biais();
    for (Index i = 0; i < b.get_row(); ++i) {
        for (Index j = 0; j < b.get_column(); ++j) {
            Scalar orig = b.get_value(i, j);

            b.get_value(i, j) = orig + epsilon; denseSingle.set_biais(b);
            Scalar loss_plus = loss_fn.calculate_loss(denseSingle.forward(inputSingle), targetSingle);

            b.get_value(i, j) = orig - epsilon; denseSingle.set_biais(b);
            Scalar loss_minus = loss_fn.calculate_loss(denseSingle.forward(inputSingle), targetSingle);

            b.get_value(i, j) = orig; denseSingle.set_biais(b);

            Scalar grad_num = (loss_plus - loss_minus) / (2.0 * epsilon);
            if (!check_gradient_relative_error(gradBiaisAna.get_value(i, j), grad_num)) biaisSingleOk = false;
        }
    }
    runner.check_values(true, biaisSingleOk, "Gradients Biais");

    bool inputSingleOk = true;
    for (Index i = 0; i < inputSingle.get_row(); ++i) {
        for (Index j = 0; j < inputSingle.get_column(); ++j) {
            Scalar orig = inputSingle.get_value(i, j);

            inputSingle.get_value(i, j) = orig + epsilon;
            Scalar loss_plus = loss_fn.calculate_loss(denseSingle.forward(inputSingle), targetSingle);

            inputSingle.get_value(i, j) = orig - epsilon;
            Scalar loss_minus = loss_fn.calculate_loss(denseSingle.forward(inputSingle), targetSingle);

            inputSingle.get_value(i, j) = orig;

            Scalar grad_num = (loss_plus - loss_minus) / (2.0 * epsilon);
            if (!check_gradient_relative_error(gradInputAna.get_value(i, j), grad_num)) inputSingleOk = false;
        }
    }
    runner.check_values(true, inputSingleOk, "Gradients Entree chainage");

    runner.section("Verification sur le reseau complet");

    Network net;

    auto ptrDense1 = std::make_unique<DenseLayer>(2, 3, gen);
    DenseLayer* dense1 = ptrDense1.get();
    net.add(std::move(ptrDense1));
    net.add(std::make_unique<Tanh>(3));

    auto ptrDense2 = std::make_unique<DenseLayer>(3, 1, gen);
    DenseLayer* dense2 = ptrDense2.get();
    net.add(std::move(ptrDense2));
    net.add(std::make_unique<Tanh>(1));

    Matrix inputNet(4, 2);
    inputNet.get_value(0,0)=0.5; inputNet.get_value(0,1)=-0.2;
    inputNet.get_value(1,0)=0.8; inputNet.get_value(1,1)=0.1;
    inputNet.get_value(2,0)=-0.4; inputNet.get_value(2,1)=0.9;
    inputNet.get_value(3,0)=0.0; inputNet.get_value(3,1)=-0.7;

    Matrix targetNet(4, 1);
    targetNet.get_value(0,0)=0.2; targetNet.get_value(1,0)=-0.5; targetNet.get_value(2,0)=0.8; targetNet.get_value(3,0)=-0.1;

    Matrix predNet = net.forward(inputNet);
    Matrix gradLossNet = loss_fn.calculate_gradients_loss(predNet, targetNet);
    net.backward(gradLossNet);

    Matrix gradWeight1Ana = dense1->get_gradWeight();
    Matrix gradBiais1Ana = dense1->get_gradBiais();
    Matrix gradWeight2Ana = dense2->get_gradWeight();
    Matrix gradBiais2Ana = dense2->get_gradBiais();

    bool weightNet1Ok = true;
    Matrix w1 = dense1->get_weight();
    for (Index i = 0; i < w1.get_row(); ++i) {
        for (Index j = 0; j < w1.get_column(); ++j) {
            Scalar orig = w1.get_value(i, j);

            w1.get_value(i, j) = orig + epsilon; dense1->set_weight(w1);
            Scalar loss_plus = loss_fn.calculate_loss(net.forward(inputNet), targetNet);

            w1.get_value(i, j) = orig - epsilon; dense1->set_weight(w1);
            Scalar loss_minus = loss_fn.calculate_loss(net.forward(inputNet), targetNet);

            w1.get_value(i, j) = orig; dense1->set_weight(w1);

            Scalar grad_num = (loss_plus - loss_minus) / (2.0 * epsilon);
            if (!check_gradient_relative_error(gradWeight1Ana.get_value(i, j), grad_num)) weightNet1Ok = false;
        }
    }
    runner.check_values(true, weightNet1Ok, "Gradients Poids Couche Dense 1");

    bool biaisNet1Ok = true;
    Matrix b1 = dense1->get_biais();
    for (Index i = 0; i < b1.get_row(); ++i) {
        for (Index j = 0; j < b1.get_column(); ++j) {
            Scalar orig = b1.get_value(i, j);

            b1.get_value(i, j) = orig + epsilon; dense1->set_biais(b1);
            Scalar loss_plus = loss_fn.calculate_loss(net.forward(inputNet), targetNet);

            b1.get_value(i, j) = orig - epsilon; dense1->set_biais(b1);
            Scalar loss_minus = loss_fn.calculate_loss(net.forward(inputNet), targetNet);

            b1.get_value(i, j) = orig; dense1->set_biais(b1);

            Scalar grad_num = (loss_plus - loss_minus) / (2.0 * epsilon);
            if (!check_gradient_relative_error(gradBiais1Ana.get_value(i, j), grad_num)) biaisNet1Ok = false;
        }
    }
    runner.check_values(true, biaisNet1Ok, "Gradients Biais Couche Dense 1");

    bool weightNet2Ok = true;
    Matrix w2 = dense2->get_weight();
    for (Index i = 0; i < w2.get_row(); ++i) {
        for (Index j = 0; j < w2.get_column(); ++j) {
            Scalar orig = w2.get_value(i, j);

            w2.get_value(i, j) = orig + epsilon; dense2->set_weight(w2);
            Scalar loss_plus = loss_fn.calculate_loss(net.forward(inputNet), targetNet);

            w2.get_value(i, j) = orig - epsilon; dense2->set_weight(w2);
            Scalar loss_minus = loss_fn.calculate_loss(net.forward(inputNet), targetNet);

            w2.get_value(i, j) = orig; dense2->set_weight(w2);

            Scalar grad_num = (loss_plus - loss_minus) / (2.0 * epsilon);
            if (!check_gradient_relative_error(gradWeight2Ana.get_value(i, j), grad_num)) weightNet2Ok = false;
        }
    }
    runner.check_values(true, weightNet2Ok, "Gradients Poids Couche Dense 2");

    bool biaisNet2Ok = true;
    Matrix b2 = dense2->get_biais();
    for (Index i = 0; i < b2.get_row(); ++i) {
        for (Index j = 0; j < b2.get_column(); ++j) {
            Scalar orig = b2.get_value(i, j);

            b2.get_value(i, j) = orig + epsilon; dense2->set_biais(b2);
            Scalar loss_plus = loss_fn.calculate_loss(net.forward(inputNet), targetNet);

            b2.get_value(i, j) = orig - epsilon; dense2->set_biais(b2);
            Scalar loss_minus = loss_fn.calculate_loss(net.forward(inputNet), targetNet);

            b2.get_value(i, j) = orig; dense2->set_biais(b2);

            Scalar grad_num = (loss_plus - loss_minus) / (2.0 * epsilon);
            if (!check_gradient_relative_error(gradBiais2Ana.get_value(i, j), grad_num)) biaisNet2Ok = false;
        }
    }
    runner.check_values(true, biaisNet2Ok, "Gradients Biais Couche Dense 2");
}