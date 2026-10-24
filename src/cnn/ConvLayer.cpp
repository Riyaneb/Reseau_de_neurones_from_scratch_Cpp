#include "cnn/ConvLayer.hpp"
#include <cassert>

ConvLayer::ConvLayer(Shape const &s, Index nKernel, Index dimKer, Random &gen, Index initChoice) : shapeIn(s), shapeOut(nKernel, s.get_height() - dimKer + 1, s.get_width() - dimKer + 1), sizeKernel(dimKer),nKer(nKernel), weight(nKernel, s.get_channel()*dimKer*dimKer), biais(1, nKernel), gradWeight(nKernel, s.get_channel()*dimKer*dimKer,0), gradBiais(1, nKernel,0), im2col(1,1) {
    assert((initChoice == 0 || initChoice == 1) && "Erreur : initChoice incorrecte");
    assert(nKernel > 0 && "Erreur : nKernel incorrecte");
    assert(dimKer > 0 && "Erreur : dimKer incorrecte (dimension négative)");
    assert(dimKer <= s.get_height() && dimKer <= s.get_width() && "Erreur : dimKer incorrecte (Ne rentre aps dans l'image)");
    if (initChoice==0) {
        init_weight_Glorot(weight, gen);
    }
    else if (initChoice==1) {
        init_weight_He(weight,gen);
    }
}

Index ConvLayer::nInput() const {
    return shapeIn.total_values();
}

Index ConvLayer::nOutput() const {
    return shapeOut.total_values();
}

PairParameters ConvLayer::get_parameters() {
    PairParameters params;
    std::pair<Matrix*, Matrix*> pair_weight;
    std::pair<Matrix*, Matrix*> pair_biais;

    pair_weight.first = &weight;
    pair_weight.second = &gradWeight;

    pair_biais.first = &biais;
    pair_biais.second = &gradBiais;

    params.push_back(pair_weight);
    params.push_back(pair_biais);

    return params;
}

void ConvLayer::set_gradients_zero() {
    gradWeight = Matrix(gradWeight.get_row(),gradWeight.get_column(),0);
    gradBiais = Matrix(gradBiais.get_row(),gradBiais.get_column(),0);
}

std::unique_ptr<Layer> ConvLayer::clone() const {
    return std::make_unique<ConvLayer>(*this);
}

Matrix ConvLayer::forward(Matrix const &in) {
    assert(in.get_column() == nInput() && "Erreur : Dimension de la matrice d'entrée incorrecte");
    Matrix outputNeural(in.get_row(),nOutput(), 0);
    Index nbPos = shapeOut.get_height() * shapeOut.get_width();
    Shape sKernel(shapeIn.get_channel(), sizeKernel, sizeKernel);
    im2col = Matrix(in.get_row() * nbPos,sKernel.total_values());
    Index cpt1 = 0;
    for (Index i = 0; i < in.get_row(); i++) {
        for (Index h = 0; h < shapeOut.get_height(); h++) {
            for (Index w = 0; w < shapeOut.get_width(); w++) {
                Index cptK = 0;
                for (Index c = 0; c < sKernel.get_channel(); c++) {
                    for (Index hk = 0; hk < sizeKernel; hk++) {
                        for (Index wk = 0; wk < sizeKernel; wk++) {
                            im2col.get_value(cpt1,cptK) = in.get_value(i,shapeIn.get_column(c,hk+h,wk+w));
                            cptK++;
                        }
                    }
                }
                cpt1++;
            }
        }
    }
    Matrix res = (im2col * weight.transpose()).broadcast(biais);
    for (Index i = 0; i < in.get_row(); i++) {
        for (Index p = 0; p < nbPos; p++) {
            for (Index k = 0; k < nKer; k++) {
                outputNeural.get_value(i,k*(nbPos) + p) = res.get_value(i*(nbPos)+p,k);
            }
        }
    }

    return outputNeural;

}

Matrix ConvLayer::backward(Matrix const &gradientOut) {
    Index nbPos = shapeOut.get_height() * shapeOut.get_width();
    assert(gradientOut.get_column() == nOutput() && gradientOut.get_row() * nbPos == im2col.get_row() && "Erreur : Dimension gradientOut");
    Matrix gradRes(nbPos*gradientOut.get_row(), nKer);
    for (Index i = 0; i < gradientOut.get_row(); i++) {
        for (Index p = 0; p < nbPos; p++) {
            for (Index k = 0; k < nKer; k++) {
                gradRes.get_value(i*(nbPos)+p,k) = gradientOut.get_value(i,k*(nbPos) + p);
            }
        }
    }
    gradWeight = gradRes.transpose() * im2col;
    gradBiais = gradRes.sum_row();
    Matrix gradBlocX = gradRes * weight;
    Matrix gradX(gradientOut.get_row(), nInput(), 0);
    Index cpt1 = 0;
    for (Index i = 0; i < gradX.get_row(); i++) {
        for (Index h = 0; h < shapeOut.get_height(); h++) {
            for (Index w = 0; w < shapeOut.get_width(); w++) {
                Index cptK = 0;
                for (Index c = 0; c < shapeIn.get_channel(); c++) {
                    for (Index hk = 0; hk < sizeKernel; hk++) {
                        for (Index wk = 0; wk < sizeKernel; wk++) {
                            gradX.get_value(i,shapeIn.get_column(c,hk+h,wk+w)) += gradBlocX.get_value(cpt1,cptK);
                            cptK++;
                        }
                    }
                }
                cpt1++;
            }
        }
    }
    return gradX;
}
