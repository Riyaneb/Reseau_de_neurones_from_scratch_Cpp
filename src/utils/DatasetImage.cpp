#include <utility>
#include "utils/DatasetImage.hpp"
#include <filesystem>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "utils/stb_image.h"
#include "utils/stb_image_resize2.h"
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <set>
#include <cstring>
#include <cstdint>

namespace fs = std::filesystem;

uint32_t DatasetImage::swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) | ((val << 8)  & 0x00FF0000) | ((val >> 8)  & 0x0000FF00) | ((val >> 24) & 0x000000FF);
}

void DatasetImage::pixel_tab_to_matrix(unsigned char *tab, Matrix &m, Index row, Shape const &s) {
    assert(m.get_column() == s.total_values() && row >= 0 && row < m.get_row() && "Erreur : Problème de dimension entre le tableau de pixel et la matrice");
    Index cpt = 0;
    for (Index h = 0; h < s.get_height(); h++) {
        for (Index w = 0; w < s.get_width(); w++) {
            for (Index c = 0; c < s.get_channel(); c++) {
                m.get_value(row, s.get_column(c, h, w)) = tab[cpt] / 255.0f;
                cpt++;
            }
        }
    }
}

void DatasetImage::folder_class(std::string const &folder, DatasetImage &dataset) {
    Index nImage = 0;

    Index actual_image = 0;

    int h;
    int w;
    int c;

    fs::path dataset_path = fs::path(DATASET_DIR) / folder;
    if (!fs::exists(dataset_path) || !fs::is_directory(dataset_path)) {
        std::cerr << "Erreur : Le dossier n'existe pas : " << dataset_path << std::endl;
        exit(EXIT_FAILURE);
    }

    for (auto const &entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_directory()) {
            dataset.class_names.push_back(entry.path().filename().string());
        }
    }

    std::sort(dataset.class_names.begin(), dataset.class_names.end());

    for (auto const &entry : fs::recursive_directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            auto extension = entry.path().extension().string();
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" || extension == ".JPG" || extension == ".PNG" || extension == ".JPEG") {
                std::string parent_folder = entry.path().parent_path().filename().string();
                auto it = std::find(dataset.class_names.begin(), dataset.class_names.end(), parent_folder);
                if (it == dataset.class_names.end()) {
                    continue;
                }
                nImage++;
            }
        }
    }


    dataset.labels = Matrix(nImage, (Index)dataset.class_names.size(),0);
    dataset.images = Matrix(nImage, dataset.shape.total_values());

    for (auto const &entry : fs::recursive_directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            auto extension = entry.path().extension().string();

            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" || extension == ".JPG" || extension == ".PNG" || extension == ".JPEG") {
                std::string parent_folder = entry.path().parent_path().filename().string();
                auto it = std::find(dataset.class_names.begin(), dataset.class_names.end(), parent_folder);
                if (it == dataset.class_names.end()) {
                    continue;
                }
                Index class_index = (Index)std::distance(dataset.class_names.begin(), it);
                auto pixels_image = stbi_load(entry.path().string().c_str(), &w, &h, &c, dataset.shape.get_channel());
                if (pixels_image == NULL) {
                    std::cerr << "Erreur : " << stbi_failure_reason() << std::endl;
                    exit(EXIT_FAILURE);
                }
                if ((Index)w != dataset.shape.get_width() || (Index)h != dataset.shape.get_height()) {
                    unsigned char *temp = nullptr;
                    temp = stbir_resize_uint8_srgb(pixels_image, w, h, 0, temp, dataset.shape.get_width(), dataset.shape.get_height(), 0, (stbir_pixel_layout)dataset.shape.get_channel());
                    stbi_image_free(pixels_image);
                    if (temp == NULL) {
                        exit(EXIT_FAILURE);
                    }
                    pixel_tab_to_matrix(temp, dataset.images, actual_image, dataset.shape);
                    free(temp);
                }
                else {
                    pixel_tab_to_matrix(pixels_image, dataset.images, actual_image, dataset.shape);
                    stbi_image_free(pixels_image);
                }
                dataset.labels.get_value(actual_image, class_index) = 1;
                actual_image++;
            }
        }

    }
    assert(actual_image == nImage && "Erreur : Problème dans le comptage d'image");

}

void DatasetImage::idx_data(std::string const &folder, DatasetImage &dataset) {
    if (dataset.shape.get_channel() != 1) {
        std::cerr << "Erreur : Problème sur le nombre de canal de couleur (1 pour les fichiers idx) " << std::endl;
        exit(EXIT_FAILURE);
    }

    fs::path idx1;
    fs::path idx3;

    fs::path dataset_path = fs::path(DATASET_DIR) / folder;

    if (!fs::exists(dataset_path) || !fs::is_directory(dataset_path)) {
        std::cerr << "Erreur : Le dossier n'existe pas : " << dataset_path << std::endl;
        exit(EXIT_FAILURE);
    }

    for (auto const &entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find("idx3") != std::string::npos) {
                idx3 = entry.path();
            }
            else if (filename.find("idx1") != std::string::npos) {
                idx1 = entry.path();
            }
        }
    }


    if (!fs::exists(idx1) ||!fs::exists(idx3)) {
        std::cerr << "Erreur : Un des deux fichiers idx n'existe pas: " << std::endl;
        exit(EXIT_FAILURE);
    }

    //Ouverture de idx1

    std::ifstream fileIdx1(idx1, std::ios::binary);
    if (!fileIdx1.is_open()) {
        std::cerr << "Erreur lors de l'ouverture du fichier." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<char> buffer1(8);
    if(!fileIdx1.read(buffer1.data(), 8)) {
        std::cerr << "Erreur : Lecture incomplète ou fin de fichier atteinte" << std::endl;
        exit(EXIT_FAILURE);
    }
    uint8_t b0 = static_cast<uint8_t>(buffer1[0]);
    uint8_t b1 = static_cast<uint8_t>(buffer1[1]);
    uint8_t b2 = static_cast<uint8_t>(buffer1[2]);
    uint8_t b3 = static_cast<uint8_t>(buffer1[3]);

    if (b0 != 0x00 || b1 != 0x00 || b2 != 0x08 || b3 != 0x01) {
        std::cerr << "Erreur : Fichier idx1 invalide" << std::endl;
        exit(EXIT_FAILURE);
    }

    uint32_t nLabels = 0;
    std::memcpy(&nLabels,&buffer1[4],sizeof(uint32_t));
    nLabels = swap_endian(nLabels);
    std::vector<Index> label_images;

    char c;
    for (Index i = 0; i < nLabels; i++) {
        if(!fileIdx1.get(c)) {
            std::cerr << "Erreur : Impossible de lire l'octet suivant" << std::endl;
            exit(EXIT_FAILURE);
        }
        Index classN = static_cast<uint8_t>(c);
        label_images.push_back(classN);
    }

    std::set<Index> unique_labels(label_images.begin(), label_images.end());
    std::set<Index>::iterator it;
    assert(nLabels > 0 && "Erreur : Aucun label trouvé");
    auto max_element = *(std::max_element(unique_labels.begin(), unique_labels.end()));

    dataset.labels = Matrix(static_cast<Index>(nLabels), max_element + 1);

    for (it = unique_labels.begin(); it != unique_labels.end(); ++it) {
        dataset.class_names.push_back(std::to_string(*it));
    }

    for (Index i = 0; i < nLabels; i++) {
        assert(label_images[i] < (max_element + 1) && label_images[i] >= 0 && "Erreur : Valeur d'un label incorrect");
        dataset.labels.get_value(i, label_images[i]) = 1;
    }

    fileIdx1.close();

    //Ouverture de idx3

    std::ifstream fileIdx3(idx3, std::ios::binary);
    if (!fileIdx3.is_open()) {
        std::cerr << "Erreur lors de l'ouverture du fichier." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<char> buffer2(16);
    if(!fileIdx3.read(buffer2.data(),16)) {
        std::cerr << "Erreur : Lecture incomplète ou fin de fichier atteinte" << std::endl;
        exit(EXIT_FAILURE);
    }

    b0 = static_cast<uint8_t>(buffer2[0]);
    b1 = static_cast<uint8_t>(buffer2[1]);
    b2 = static_cast<uint8_t>(buffer2[2]);
    b3 = static_cast<uint8_t>(buffer2[3]);

    if (b0 != 0x00 || b1 != 0x00 || b2 != 0x08 || b3 != 0x03) {
        std::cerr << "Erreur : Fichier idx3 invalide" << std::endl;
        exit(EXIT_FAILURE);
    }

    uint32_t nImage = 0;
    std::memcpy(&nImage,&buffer2[4],sizeof(uint32_t));
    nImage = swap_endian(nImage);

    uint32_t nRow = 0;
    std::memcpy(&nRow,&buffer2[8],sizeof(uint32_t));
    nRow = swap_endian(nRow);

    uint32_t nCol = 0;
    std::memcpy(&nCol,&buffer2[12],sizeof(uint32_t));
    nCol = swap_endian(nCol);

    if (static_cast<Index>(nImage) != nLabels || static_cast<Index>(nRow) != dataset.shape.get_height() || static_cast<Index>(nCol) != dataset.shape.get_width()) {
        std::cerr << "Erreur : Dimension invalide dans le fichier idx3" << std::endl;
        exit(EXIT_FAILURE);
    }

    dataset.images = Matrix(static_cast<Index>(nImage), dataset.shape.total_values());
    std::vector<unsigned char> temp(dataset.shape.total_values());


    for (Index i = 0; i < static_cast<Index>(nImage); i++) {
        if (!fileIdx3.read(reinterpret_cast<char*>(temp.data()), dataset.get_shape().total_values())) {
            std::cerr << "Erreur : Lecture incomplète ou fin de fichier atteinte" << std::endl;
            exit(EXIT_FAILURE);
        }
        pixel_tab_to_matrix(temp.data(), dataset.images,i,dataset.shape);
    }

    fileIdx3.close();
}

Shape const &DatasetImage::get_shape() const {
    return shape;
}

Matrix const &DatasetImage::get_images() const {
    return images;
}

Matrix const &DatasetImage::get_labels() const {
    return labels;
}

std::vector<std::string> const &DatasetImage::get_class_names() const {
    return class_names;
}

Index DatasetImage::get_nClass() const {
    return (Index)class_names.size();
}

Index DatasetImage::get_nImages() const {
    return images.get_row();
}


DatasetImage::DatasetImage(Shape const &s, std::string p, int datasetType, std::uint32_t seed) : shape(s), path(p), images(1,1), labels(1,1), gen(seed) {
    assert(datasetType >= 0 && datasetType <= 3 && "Erreur : Le type du dataset est incorrect");
    switch (datasetType) {
        case 0:
            folder_class(p,*this);
            break;
        case 1:
            idx_data(p,*this);
            break;
        case 2:
            break;

        case 3:
            break;
        default:
            std::cerr << "Erreur : Type du dataset incorrect" << std::endl;
            exit(EXIT_FAILURE);
    }
}

void DatasetImage::shuffle() {
    Index n = get_nImages();
    std::vector<Index> tab(n);

    for (Index i = 0; i < n; ++i) {
        tab[i] = i;
    }

    for (Index i = n-1; i>0; --i) {
        Index j = gen.randomizeInt(0,i);
        std::swap(tab[i], tab[j]);
    }

    Matrix tempIm(images.get_row(), images.get_column(), 0);
    Matrix temLab(labels.get_row(), labels.get_column(), 0);

    for (Index i = 0; i < images.get_row(); ++i) {
        for (Index j = 0; j < images.get_column(); ++j) {
            tempIm.get_value(i,j) = images.get_value(tab[i],j);
        }
    }

    for (Index i = 0; i < labels.get_row(); ++i) {
        for (Index j = 0; j < labels.get_column(); ++j) {
            temLab.get_value(i,j) = labels.get_value(tab[i],j);
        }
    }

    images = tempIm;
    labels = temLab;
}

std::pair<Matrix, Matrix> DatasetImage::get_batch(Index begin_row, Index size) const {
    assert(begin_row < images.get_row() && size > 0 && images.get_row() == labels.get_row() && "Erreur : Dimension incorrecte");
    Index sizeDynam = std::min(size,images.get_row() - begin_row);

    Matrix resIm(sizeDynam, images.get_column());
    Matrix resLab(sizeDynam, labels.get_column());

    Index cpt = 0;

    for (Index i = begin_row; i < begin_row + sizeDynam; ++i) {
        for (Index j = 0; j < images.get_column(); ++j) {
            resIm.get_value(cpt,j) = images.get_value(i,j);
        }
        for (Index j = 0; j < labels.get_column(); ++j) {
            resLab.get_value(cpt,j) = labels.get_value(i,j);
        }
        cpt++;
    }

    std::pair<Matrix, Matrix> res = std::make_pair(std::move(resIm), std::move(resLab));
    return res;

}
