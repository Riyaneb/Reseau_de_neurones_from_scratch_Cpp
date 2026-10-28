#ifndef RESEAUDENEURONEFROMSCRATCH_PREDICTIONCNN_HPP
#define RESEAUDENEURONEFROMSCRATCH_PREDICTIONCNN_HPP

#include <utility>

#include "nn/Shape.hpp"


struct Answer_gui {
    Answer_gui(std::vector<std::string> const &labels, std::vector<Scalar> const &percentages, std::string title, Scalar sc, bool a, Index id, std::string res) : class_names(labels), score(percentages), answer_title(std::move(title)), answer_score(sc),answer(a), answer_index(id), real_result(std::move(res)){}
    std::vector<std::string> class_names;
    std::vector<Scalar> score;
    std::string answer_title;
    Scalar answer_score;
    bool answer;
    Index answer_index;
    std::string real_result;
};

#endif //RESEAUDENEURONEFROMSCRATCH_PREDICTIONCNN_HPP
