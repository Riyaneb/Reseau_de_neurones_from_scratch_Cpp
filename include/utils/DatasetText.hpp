#ifndef RESEAUDENEURONEFROMSCRATCH_DATASETTEXT_HPP
#define RESEAUDENEURONEFROMSCRATCH_DATASETTEXT_HPP

#include "math/Matrix.hpp"
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <cassert>

#include "math/Random.hpp"
#include <cstdint>
#define VOCAB_SIZE 256


class DatasetText {
public:
    DatasetText(std::string path_file, Index type, std::uint32_t seed = 0);
    std::pair<Matrix, Matrix> get_batch(Index batch_size, Index txt_size, bool in_gpu=false) ;
    std::pair<Matrix, Matrix> get_batch_in_order(Index num_batch, Index batch_size, Index txt_size, bool in_gpu=false) const;
    std::string get_line(Index i) const;

    Index get_nb_batch(Index batch_size, Index txt_size) const {
        assert(batch_size > 0 && txt_size > 0 && "Erreur : Paramètres incorrecte");
        return (text.size()-1) / (batch_size*txt_size);
    }

    std::size_t get_size_text() const {
        return text.size();
    }

    Index get_vocab_size() const {
        return vocabSize;
    }

    std::size_t get_nb_lines() const {
        return line_start.size();
    }

    static std::vector<uint8_t> encode(std::string const &txt) {
        return std::vector<uint8_t>(txt.begin(), txt.end());
    }

    static std::string decode(std::vector<uint8_t> const &txt) {
        return std::string(txt.begin(), txt.end());
    }

private:
    static void raw_txt(std::string const &path_file, DatasetText &dataset);

    static void txt_tab_to_matrix(const uint8_t* tab, Matrix &m, std::size_t row, std::size_t txt_size, std::size_t begin_txt, bool is_x);

    std::string path;
    std::vector<uint8_t> text;
    std::vector<std::size_t> line_start;
    std::vector<uint8_t> answer_mask;
    Random gen;
    static constexpr Index vocabSize = VOCAB_SIZE;
};

#endif //RESEAUDENEURONEFROMSCRATCH_DATASETTEXT_HPP
