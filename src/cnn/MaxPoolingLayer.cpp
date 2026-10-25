#include "cnn/MaxPoolingLayer.hpp"

MaxPoolingLayer::MaxPoolingLayer(Shape const &s, Index dimKer) : shapeIn(s), shapeOut(s.get_channel(), s.get_height() / dimKer, s.get_width() / dimKer), sizeKernel(dimKer), posMaxim(1,1) {
    assert(dimKer > 0 && "Erreur : dimKer incorrecte (dimension négative)");
    assert(dimKer <= s.get_height() && dimKer <= s.get_width() && "Erreur : dimKer incorrecte (Ne rentre pas dans l'image)");
}

Index MaxPoolingLayer::nInput() const {
    return shapeIn.total_values();
}

Index MaxPoolingLayer::nOutput() const {
    return shapeOut.total_values();
}

std::unique_ptr<Layer> MaxPoolingLayer::clone() const {
    return std::make_unique<MaxPoolingLayer>(*this);
}

std::pair<Scalar, Index> MaxPoolingLayer::max(std::vector<std::pair<Scalar, Index>> const &tab) {
    assert(!tab.empty() && "Tableau vide");
    std::pair<Scalar, Index> max = tab[0];
    std::vector<std::pair<Scalar, Index>>::const_iterator it;
    for (it = tab.begin(); it != tab.end(); ++it) {
        if ((*it).first > max.first) {
            max = *it;
        }
    }
    return max;
}


Matrix MaxPoolingLayer::forward(Matrix const &in) {
    assert(in.get_column() == nInput() && "Erreur : Dimension de l'entrée incorrecte");
    Matrix result(in.get_row(), nOutput());
    posMaxim = Matrix(in.get_row(), nOutput());
    for (Index i = 0; i < in.get_row(); i++) {
        Index cpt = 0;
        for (Index c = 0; c < shapeIn.get_channel(); c++){
            for (Index h = 0; h < shapeOut.get_height() * sizeKernel; h+=sizeKernel) {
                for (Index w = 0; w < shapeOut.get_width() * sizeKernel; w+=sizeKernel) {
                    std::vector<std::pair<Scalar, Index>> temp;
                    for (Index hk = 0; hk < sizeKernel; hk++) {
                        for (Index wk = 0; wk < sizeKernel; wk++) {
                            temp.emplace_back(in.get_value(i,shapeIn.get_column(c,hk+h,wk+w)),shapeIn.get_column(c,hk+h,wk+w));
                        }
                    }
                    auto p = max(temp);
                    result.get_value(i,cpt) = p.first;
                    posMaxim.get_value(i,cpt) = p.second;
                    cpt++;
                }
            }
        }
    }
    return result;
}

Matrix MaxPoolingLayer::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == nOutput() && gradientOut.get_row() == posMaxim.get_row() && "Erreur : Dimension gradientOut incorrecte");
    Matrix gradX(gradientOut.get_row(), nInput(), 0);
    for (Index i = 0; i < gradientOut.get_row(); i++) {
        for (Index j = 0; j < posMaxim.get_column(); j++) {
            gradX.get_value(i,(Index)posMaxim.get_value(i,j)) = gradientOut.get_value(i,j);
        }
    }
    return gradX;
}
