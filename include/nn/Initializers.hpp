#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_INITIALIZERS_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_INITIALIZERS_HPP

#include <cmath>

#include "Shape.hpp"
#include "math/Random.hpp"
#include "math/Matrix.hpp"

void init_weight_Glorot(Matrix &weight, Random &gen);
void init_weight_He(Matrix &weight, Random &gen);
void init_weight_He_CNN(Matrix &weight, Random &gen);
void init_weight_Glorot_CNN(Matrix &weight, Random &gen, Index dimk);

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_INITIALIZERS_HPP
