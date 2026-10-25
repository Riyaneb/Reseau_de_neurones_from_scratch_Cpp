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

namespace fs = std::filesystem;

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
