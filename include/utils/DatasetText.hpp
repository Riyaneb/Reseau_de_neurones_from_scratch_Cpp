#ifndef RESEAUDENEURONEFROMSCRATCH_DATASETTEXT_HPP
#define RESEAUDENEURONEFROMSCRATCH_DATASETTEXT_HPP

#include "math/Matrix.hpp"
#include "nn/Shape.hpp"
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "math/Random.hpp"
#include <cstdint>
#define VOCAB_SIZE 256


class DatasetText {
public:
    DatasetText(std::string path_file, Index type, std::uint32_t seed = 0);

private:
    static void raw_txt(std::string const &path_file, DatasetText &dataset);

    std::string path;
    std::vector<uint8_t> text;
    std::vector<std::size_t> line_start;
    std::vector<uint8_t> answer_mask;
    Random gen;
    static constexpr Index vocabSize = VOCAB_SIZE;
};

#endif //RESEAUDENEURONEFROMSCRATCH_DATASETTEXT_HPP
