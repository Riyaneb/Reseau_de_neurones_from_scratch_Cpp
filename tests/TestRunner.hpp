#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_TESTRUNNER_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_TESTRUNNER_HPP

#include <iostream>
#include <cmath>
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



#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_TESTRUNNER_HPP
