#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_SERIALIZER_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_SERIALIZER_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <iomanip>
#include "nn/Network.hpp"
#include "nn/DenseLayer.hpp"

bool save(Network &net, std::string const &filename);
bool load(Network &net, std::string const &filename, bool in_gpu = false);

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_SERIALIZER_HPP
