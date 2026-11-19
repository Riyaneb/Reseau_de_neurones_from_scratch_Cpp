#include <utility>

#include "utils/DatasetText.hpp"
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;


void DatasetText::txt_tab_to_matrix(const uint8_t *tab, Matrix &m, std::size_t row, std::size_t txt_size, std::size_t begin_txt, bool is_x) {
    for (Index i = 0; i < txt_size; i++) {
        if (is_x) {
            m.get_value(row+i, 0) = tab[begin_txt+i];
        }
        else {
            m.get_value(row+i,tab[begin_txt + i]) = 1;
        }
    }
}

void DatasetText::raw_txt(std::string const &path_file, DatasetText &dataset) {

    fs::path dataset_path = fs::path(DATASET_TEXT_DIR) / path_file;

    if (!fs::exists(dataset_path) || !fs::is_directory(dataset_path)) {
        std::cerr << "Erreur : Le dossier n'existe pas : " << dataset_path << std::endl;
        exit(EXIT_FAILURE);
    }

    Index cpt = 0;
    for (auto const &entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            cpt++;
        }
    }

    if (cpt != 1) {
        std::cerr << "Erreur : Le dossier " << dataset_path << " doit contenir uniquement un seul fichier txt"<< std::endl;
        exit(EXIT_FAILURE);
    }

    for (auto const &entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            auto extension = entry.path().extension().string();
            if (extension == ".txt") {
                std::size_t size_file = entry.file_size();
                dataset.text.resize(size_file);
                if (size_file > 0) {
                    std::ifstream file(entry.path().string(), std::ios::binary);
                    if (!file) {
                        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << entry.path() << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    file.read(reinterpret_cast<char*>(dataset.text.data()), size_file);
                    if (!file) {
                        std::cerr << "Erreur : Lecture incomplète du fichier " << entry.path() << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    dataset.line_start.clear();
                    dataset.line_start.push_back(0);
                    for (std::size_t i = 0; i < size_file; i++) {
                        if (dataset.text[i] == '\n') {
                            if (i+1 < size_file) {
                                dataset.line_start.push_back(i+1);
                            }
                        }
                    }
                }
                else {
                    std::cerr << "Erreur : Le fichier " << entry.path() << " est vide"<< std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            else {
                std::cerr << "Erreur : Le dossier " << dataset_path << " doit contenir uniquement un seul fichier txt"<< std::endl;
                exit(EXIT_FAILURE);
            }

        }
    }

}

DatasetText::DatasetText(std::string path_file, Index type, std::uint32_t seed) : path(std::move(path_file)), text(0), line_start(0), answer_mask(0), gen(seed) {
    assert(type >= 0 && type <= 1 && "Erreur : Le type du dataset est incorrect");
    switch (type) {
        case 0:
            raw_txt(path, *this);
            break;
        case 1:
            break;
        default:
            break;
    }

}

std::pair<Matrix, Matrix> DatasetText::get_batch(Index batch_size, Index txt_size, bool in_gpu) {
    assert(batch_size > 0 && txt_size > 0 && "Erreur : Paramètres incorrecte");
    assert(txt_size < text.size() && "Erreur : La longueur du texte dépasse la longueur du fichier");
    Matrix x(batch_size*txt_size, 1), y(batch_size*txt_size, vocabSize);

    for (std::size_t i = 0; i < batch_size; i++) {
        std::size_t begin_txt = gen.randomizeBigInt(0, text.size()-1-txt_size);
        txt_tab_to_matrix(text.data(), x, txt_size*i, txt_size, begin_txt, true);
        txt_tab_to_matrix(text.data(), y, txt_size*i, txt_size, begin_txt+1, false);
    }

    if (in_gpu) {
        x.to_gpu();
        y.to_gpu();
    }

    return std::make_pair(std::move(x),std::move(y));
}

std::pair<Matrix, Matrix> DatasetText::get_batch_in_order(Index num_batch, Index batch_size, Index txt_size, bool in_gpu) const {
    assert(batch_size > 0 && txt_size > 0 && "Erreur : Paramètres incorrecte");
    assert(txt_size < text.size() && "Erreur : La longueur du texte dépasse la longueur du fichier");
    assert(num_batch >= 0 && static_cast<size_t>(num_batch+1)*batch_size*txt_size < text.size() && "Erreur : num_batch incorrect");
    Matrix x(batch_size*txt_size, 1), y(batch_size*txt_size, vocabSize);

    for (std::size_t i = 0; i < batch_size; i++) {
        std::size_t begin_txt = (static_cast<size_t>(num_batch)*batch_size+i) * txt_size;
        txt_tab_to_matrix(text.data(), x, txt_size*i, txt_size, begin_txt, true);
        txt_tab_to_matrix(text.data(), y, txt_size*i, txt_size, begin_txt+1, false);
    }

    if (in_gpu) {
        x.to_gpu();
        y.to_gpu();
    }

    return std::make_pair(std::move(x),std::move(y));
}


std::string DatasetText::get_line(Index i) const {
    assert(i < line_start.size() && i >= 0 && "Erreur : Numero de ligne incorrect");
    if (i+1 == line_start.size()) {
        return std::string(text.begin()+line_start[i], text.end());
    }
    return std::string(text.begin()+line_start[i], text.begin()+line_start[i+1]-1);
}
