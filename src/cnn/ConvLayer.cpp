#include "cnn/ConvLayer.hpp"
#include <cassert>
#include <iostream>
#include <cstdlib>
#include "utils/Timer.hpp"
#ifdef USE_CUDA
#include "gpu/Gpu_Convolution.hpp"
#endif

ConvLayer::ConvLayer(Shape const &s, Index nKernel, Index dimKer, Random &gen, Index initChoice) : shapeIn(s), shapeOut(nKernel, s.get_height() - dimKer + 1, s.get_width() - dimKer + 1), sizeKernel(dimKer),nKer(nKernel), weight(nKernel, s.get_channel()*dimKer*dimKer), biais(1, nKernel), gradWeight(nKernel, s.get_channel()*dimKer*dimKer,0), gradBiais(1, nKernel,0), im2col(1,1) {
    assert((initChoice == 0 || initChoice == 1) && "Erreur : initChoice incorrecte");
    assert(nKernel > 0 && "Erreur : nKernel incorrecte");
    assert(dimKer > 0 && "Erreur : dimKer incorrecte (dimension négative)");
    assert(dimKer <= s.get_height() && dimKer <= s.get_width() && "Erreur : dimKer incorrecte (Ne rentre pas dans l'image)");
    if (initChoice==0) {
        init_weight_Glorot_CNN(weight, gen, dimKer);
    }
    else if (initChoice==1) {
        init_weight_He_CNN(weight,gen);
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
    gradWeight.fill(0);
    gradBiais.fill(0);
}

std::unique_ptr<Layer> ConvLayer::clone() const {
    return std::make_unique<ConvLayer>(*this);
}

Matrix ConvLayer::forward(Matrix const &in) {
    assert(in.get_column() == nInput() && "Erreur : Dimension de la matrice d'entrée incorrecte");
    Matrix outputNeural(in.get_row(),nOutput(), 0, in.get_in_gpu());
    Index nbPos = shapeOut.get_height() * shapeOut.get_width();
    Shape sKernel(shapeIn.get_channel(), sizeKernel, sizeKernel);

    im2col = Matrix(in.get_row() * nbPos,sKernel.total_values(), 0, in.get_in_gpu());
    if (in.get_in_gpu()) {
        #ifdef USE_CUDA
            forward_im2col_gpu(in.get_gpu_ptr(), im2col.get_gpu_ptr(), nbPos, im2col.get_row(), im2col.get_column(), shapeIn.get_height(), shapeIn.get_width(), nInput(), shapeOut.get_width(), sizeKernel);
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for schedule(dynamic,1)
        for (Index i = 0; i < in.get_row(); i++) {
            for (Index h = 0; h < shapeOut.get_height(); h++) {
                for (Index w = 0; w < shapeOut.get_width(); w++) {
                    Index cptK = 0;
                    for (Index c = 0; c < sKernel.get_channel(); c++) {
                        for (Index hk = 0; hk < sizeKernel; hk++) {
                            for (Index wk = 0; wk < sizeKernel; wk++) {
                                im2col.get_value(i*(nbPos) + h*shapeOut.get_width() + w,cptK) = in.get_value(i,shapeIn.get_column(c,hk+h,wk+w));
                                cptK++;
                            }
                        }
                    }
                }
            }
        }
    }

    Matrix res = im2col * weight.transpose();

    res.broadcast(biais);

    if (outputNeural.get_in_gpu()) {
        #ifdef USE_CUDA
            forward_output_gpu(res.get_gpu_ptr(), outputNeural.get_gpu_ptr(), nbPos, nKer, in.get_row());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for schedule(dynamic,1)
        for (Index i = 0; i < in.get_row(); i++) {
            for (Index p = 0; p < nbPos; p++) {
                for (Index k = 0; k < nKer; k++) {
                    outputNeural.get_value(i,k*(nbPos) + p) = res.get_value(i*(nbPos)+p,k);
                }
            }
        }
    }
    return outputNeural;

}

Matrix ConvLayer::backward(Matrix const &gradientOut) {
    Index nbPos = shapeOut.get_height() * shapeOut.get_width();
    assert(gradientOut.get_column() == nOutput() && gradientOut.get_row() * nbPos == im2col.get_row() && "Erreur : Dimension gradientOut");
    assert(gradientOut.get_in_gpu() == im2col.get_in_gpu() && "Erreur : gradientOut et im2col ne sont pas dans le même device");

    Matrix gradRes(nbPos*gradientOut.get_row(), nKer, 0, gradientOut.get_in_gpu());

    if (gradRes.get_in_gpu()) {
        #ifdef USE_CUDA
            backward_output_gpu(gradientOut.get_gpu_ptr(), gradRes.get_gpu_ptr(), nbPos, nKer, gradientOut.get_row());
        #else
            std::cerr << "Erreur : CUDA non disponible" << std::endl;
            exit(EXIT_FAILURE);
        #endif
    }
    else {
        #pragma omp parallel for schedule(dynamic,1)
        for (Index i = 0; i < gradientOut.get_row(); i++) {
            for (Index p = 0; p < nbPos; p++) {
                for (Index k = 0; k < nKer; k++) {
                    gradRes.get_value(i*(nbPos)+p,k) = gradientOut.get_value(i,k*(nbPos) + p);
                }
            }
        }
    }


    gradWeight = gradRes.transpose() * im2col;

    gradBiais = gradRes.sum_row();

    if (need_grad_input) {
        Matrix gradBlocX = gradRes * weight;
        Matrix gradX(gradientOut.get_row(), nInput(), 0, gradientOut.get_in_gpu());
        if (gradX.get_in_gpu()) {
            #ifdef USE_CUDA
                backward_col2im_gpu(gradBlocX.get_gpu_ptr(), gradX.get_gpu_ptr(), nbPos, im2col.get_row(), im2col.get_column(), shapeIn.get_height(), shapeIn.get_width(), nInput(), shapeOut.get_width(), sizeKernel);
            #else
                std::cerr << "Erreur : CUDA non disponible" << std::endl;
                exit(EXIT_FAILURE);
            #endif
        }
        else {
            #pragma omp parallel for schedule(dynamic,1)
            for (Index i = 0; i < gradX.get_row(); i++) {
                for (Index h = 0; h < shapeOut.get_height(); h++) {
                    for (Index w = 0; w < shapeOut.get_width(); w++) {
                        Index cptK = 0;
                        for (Index c = 0; c < shapeIn.get_channel(); c++) {
                            for (Index hk = 0; hk < sizeKernel; hk++) {
                                for (Index wk = 0; wk < sizeKernel; wk++) {
                                    gradX.get_value(i,shapeIn.get_column(c,hk+h,wk+w)) += gradBlocX.get_value(i*(nbPos) + h*shapeOut.get_width() + w,cptK);
                                    cptK++;
                                }
                            }
                        }
                    }
                }
            }
        }
        return gradX;
    }
    return Matrix(1,1);
}