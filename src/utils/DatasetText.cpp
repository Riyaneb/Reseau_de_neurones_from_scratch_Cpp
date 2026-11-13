#include <utility>

#include "utils/DatasetText.hpp"
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <cassert>

namespace fs = std::filesystem;


void DatasetText::raw_txt(std::string const &path_file, DatasetText &dataset) {

    fs::path dataset_path = fs::path(DATASET_DIR) / path_file;

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
