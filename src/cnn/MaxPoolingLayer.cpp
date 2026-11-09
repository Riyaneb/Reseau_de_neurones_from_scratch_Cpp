#include "cnn/MaxPoolingLayer.hpp"
#include <omp.h>

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


Matrix MaxPoolingLayer::forward(Matrix const &in) {
    assert(in.get_column() == nInput() && "Erreur : Dimension de l'entrée incorrecte");
    Matrix result(in.get_row(), nOutput(), 0, in.get_in_gpu());
    posMaxim = Matrix(in.get_row(), nOutput(), 0, in.get_in_gpu());

    if (in.get_in_gpu()) {
        
    }
    else {
#pragma omp parallel for schedule(dynamic,1)
        for (Index i = 0; i < in.get_row(); i++) {
            Scalar maxi;
            Index maxi_id;
            Index cpt = 0;
            for (Index c = 0; c < shapeIn.get_channel(); c++){
                for (Index h = 0; h < shapeOut.get_height() * sizeKernel; h+=sizeKernel) {
                    for (Index w = 0; w < shapeOut.get_width() * sizeKernel; w+=sizeKernel) {
                        maxi = in.get_value(i,shapeIn.get_column(c,h,w));
                        maxi_id = shapeIn.get_column(c,h,w);
                        for (Index hk = 0; hk < sizeKernel; hk++) {
                            for (Index wk = 0; wk < sizeKernel; wk++) {
                                Scalar temp = in.get_value(i,shapeIn.get_column(c,hk+h,wk+w));
                                if (temp > maxi) {
                                    maxi = temp;
                                    maxi_id = shapeIn.get_column(c,hk+h,wk+w);
                                }
                            }
                        }
                        result.get_value(i,cpt) = maxi;
                        posMaxim.get_value(i,cpt) = maxi_id;
                        cpt++;
                    }
                }
            }
        }
    }
    return result;
}

Matrix MaxPoolingLayer::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == nOutput() && gradientOut.get_row() == posMaxim.get_row() && "Erreur : Dimension gradientOut incorrecte");
    Matrix gradX(gradientOut.get_row(), nInput(), 0, gradientOut.get_in_gpu());
    #pragma omp parallel for schedule(dynamic,1)
    for (Index i = 0; i < gradientOut.get_row(); i++) {
        for (Index j = 0; j < posMaxim.get_column(); j++) {
            gradX.get_value(i,(Index)posMaxim.get_value(i,j)) = gradientOut.get_value(i,j);
        }
    }
    return gradX;
}
