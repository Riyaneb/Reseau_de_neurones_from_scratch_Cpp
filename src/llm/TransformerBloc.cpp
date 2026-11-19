#include "llm/TransformerBloc.hpp"
#include <iterator>

TransformerBloc::TransformerBloc(Index d, Index h, Random &gen) : dimEmbedding(d), nHead(h), context(0), norm1(d), attention(d,h,gen), norm2(d), dense1(d,4*d,gen,2), geLu(4*d), dense2(4*d, d, gen,2) {}

Index TransformerBloc::nInput() const {
    return dimEmbedding;
}

Index TransformerBloc::nOutput() const {
    return dimEmbedding;
}

void TransformerBloc::set_gradients_zero() {
    norm1.set_gradients_zero();
    norm2.set_gradients_zero();
    attention.set_gradients_zero();
    dense1.set_gradients_zero();
    dense2.set_gradients_zero();
}

PairParameters TransformerBloc::get_parameters() {
    auto param1 = norm1.get_parameters();
    auto param2 = attention.get_parameters();
    auto param3 = norm2.get_parameters();
    auto param4 = dense1.get_parameters();
    auto param5 = dense2.get_parameters();

    PairParameters parametersFinal;
    parametersFinal.reserve(param1.size() + param2.size() + param3.size() + param4.size() + param5.size());

    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(param1.begin()), std::make_move_iterator(param1.end()));
    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(param2.begin()), std::make_move_iterator(param2.end()));
    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(param3.begin()), std::make_move_iterator(param3.end()));
    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(param4.begin()), std::make_move_iterator(param4.end()));
    parametersFinal.insert(parametersFinal.end(), std::make_move_iterator(param5.begin()), std::make_move_iterator(param5.end()));

    return parametersFinal;
}

std::unique_ptr<Layer> TransformerBloc::clone() const {
    return std::make_unique<TransformerBloc>(*this);
}

Matrix TransformerBloc::forward(Matrix const &in) {
    assert(in.get_column() == dimEmbedding && "Erreur : Dimension de in incorrect");
    Matrix temp1 = norm1.forward(in);
    temp1 = attention.forward(temp1);
    Matrix residu1 = in + temp1;
    Matrix temp2 = norm2.forward(residu1);
    temp2 = dense1.forward(temp2);
    temp2 = geLu.forward(temp2);
    temp2 = dense2.forward(temp2);
    return residu1 + temp2;
}

Matrix TransformerBloc::backward(Matrix const &gradientOut) {
    assert(gradientOut.get_column() == dimEmbedding && "Erreur : Dimension de gradientOut incorrect");
    Matrix temp1 = dense2.backward(gradientOut);
    temp1 = geLu.backward(temp1);
    temp1 = dense1.backward(temp1);
    temp1 = norm2.backward(temp1);
    Matrix gradResidu1 = temp1 + gradientOut;
    Matrix temp2 = attention.backward(gradResidu1);
    temp2 = norm1.backward(temp2);
    return gradResidu1 + temp2;
}

