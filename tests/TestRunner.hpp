#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_TESTRUNNER_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_TESTRUNNER_HPP

#include <iostream>
#include <cmath>
#include <sstream>
#include <string>

#include "math/Types.hpp"
#include "math/Matrix.hpp"

class TestRunner {
public:
    TestRunner(): nb_fail(0), nb_succes(0) {}

    void check_test(bool result, std::string const &name) {
        if (result) {
            ++nb_succes;
            std::cout << "[OK] " << name << std::endl;
        }
        else {
            ++nb_fail;
            std::cout << "[ECHEC] " << name << std::endl;
        }
    }

    void check_values(int value_expected, int value_obtained, std::string const &name) {
        bool succes(value_expected == value_obtained);
        check_test(succes, name);
        if (!succes) {
            std::cout << "attendu " << value_expected << ", obtenu : " << value_obtained << std::endl;
        }
    }

    void check_values(Scalar value_expected, Scalar value_obtained, std::string const &name, double epsilon = 1e-2) {
        bool succes(std::abs(value_obtained - value_expected) < epsilon);
        check_test(succes, name);
        if (!succes) {
            std::cout << "attendu " << value_expected << ", obtenu : " << value_obtained << std::endl;
        }
    }

    void check_values(bool value_expected, bool value_obtained, std::string const &name) {
        bool succes(value_expected == value_obtained);
        check_test(succes, name);
        if (!succes) {
            std::cout << "attendu " << value_expected << ", obtenu : " << value_obtained << std::endl;
        }
    }

    void section(std::string const &name) const {
        std::cout << std::endl << "======" << name << "======" << std::endl;
    }

    void bilan() const {
        std::cout << "Bilan : " << nb_succes << " succès, " << nb_fail << " échoués" << std::endl;
    }

    int return_test() const {
        return nb_fail;
    }

private:
    int nb_fail;
    int nb_succes;
};

inline void comparaison_matrix(TestRunner &runner, Matrix const &result, Matrix const &expected, std::string const &nom) {
    std::ostringstream nCo;
    nCo << nom << " : Test dimension colonne";
    std::ostringstream nRow;
    nRow << nom << " : Test dimension ligne";
    runner.check_values(expected.get_row(), result.get_row(), nRow.str());
    runner.check_values(expected.get_column(),result.get_column(),nCo.str());
    if (expected.get_row() != result.get_row() || expected.get_column() != result.get_column()) {
        return;
    }

    std::ostringstream matExpectedStr;
    std::ostringstream matResultStr;

    for (Index i = 0; i < expected.get_row(); i++) {
        for (Index j = 0; j < expected.get_column(); j++) {
            std::ostringstream casee;
            casee << nom << " cellule (" << i << "," << j << ")";
            runner.check_values(expected.get_value(i,j), result.get_value(i,j), casee.str());
            matExpectedStr << expected.get_value(i,j) << " ";
            matResultStr<< result.get_value(i,j) << " ";
        }
        matExpectedStr << std::endl;
        matResultStr << std::endl;
    }

    std::cout << "Matrice attendu :\n\n" << matExpectedStr.str() << "\n\nMatrice obtenu :\n\n" << matResultStr.str() << std::endl;
}

inline void comparaison_matrix_big(TestRunner &runner, Matrix const &result, Matrix const &expected, std::string const &nom) {
    std::ostringstream nCo;
    nCo << nom << " : Test dimension colonne";
    std::ostringstream nRow;
    nRow << nom << " : Test dimension ligne";
    runner.check_values(expected.get_row(), result.get_row(), nRow.str());
    runner.check_values(expected.get_column(),result.get_column(),nCo.str());
    if (expected.get_row() != result.get_row() || expected.get_column() != result.get_column()) {
        return;
    }

    Index nb_error = 0;
    for (Index i = 0; i < expected.get_row(); i++) {
        for (Index j = 0; j < expected.get_column(); j++) {
            if (result.get_value(i,j) != expected.get_value(i,j)) {
                ++nb_error;
                std::cout << "[ERROR] case(" << i << "," << j << ") : Attendu " << expected.get_value(i,j) << ", Obtenu " << result.get_value(i,j) << std::endl;
            }
        }
    }
    runner.check_values(0, nb_error, nom);
}

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_TESTRUNNER_HPP
