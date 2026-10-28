#ifndef RESEAUDENEURONEFROMSCRATCH_DATASETIMAGE_HPP
#define RESEAUDENEURONEFROMSCRATCH_DATASETIMAGE_HPP


#include "math/Matrix.hpp"
#include "nn/Shape.hpp"
#include <string>
#include <utility>
#include <vector>

#include "math/Random.hpp"
#include <cstdint>

class DatasetImage {
public:
    DatasetImage(Shape const &s, std::string p, int datasetType, std::uint32_t seed = 0);
    Shape const &get_shape() const;
    Matrix const &get_images() const;
    Matrix const &get_labels() const;
    std::vector<std::string> const &get_class_names() const;
    Index get_nClass() const;
    Index get_nImages() const;

    void shuffle();
    std::pair<Matrix, Matrix> get_batch(Index row, Index size) const;

private:
    static void folder_class(std::string const &folder, DatasetImage &dataset);
    static void label_name(std::string const &folder, DatasetImage &dataset) {};
    static void csv_link(std::string const &folder, DatasetImage &dataset){};
    static void idx_data(std::string const &folder, DatasetImage &dataset);

    static void pixel_tab_to_matrix(unsigned char* tab, Matrix &m, Index row, Shape const &s);
    static uint32_t swap_endian(uint32_t val);

    Shape shape;
    std::string path;
    Matrix images;
    Matrix labels;
    std::vector<std::string> class_names;
    Random gen;
};


#endif //RESEAUDENEURONEFROMSCRATCH_DATASETIMAGE_HPP
