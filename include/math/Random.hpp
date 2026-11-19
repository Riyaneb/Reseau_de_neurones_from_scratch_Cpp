#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_RANDOM_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_RANDOM_HPP
#include <random>
#include "math/Types.hpp"

class Random {
public:
    Random(std::uint32_t seed = 0) {
        engine.seed(seed);
    };
    void changeSeed(std::uint32_t seed);
    Scalar randomizeUniform(int a, int b);
    Scalar randomizeNormal(double mean, double stddev);
    Index randomizeInt(int a, int b);
    std::size_t randomizeBigInt(std::size_t a, std::size_t b);
private:
    std::mt19937 engine;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_RANDOM_HPP
