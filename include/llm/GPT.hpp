#ifndef RESEAUDENEURONEFROMSCRATCH_GPT_HPP
#define RESEAUDENEURONEFROMSCRATCH_GPT_HPP

#include "llm/EmbeddingLayer.hpp"
#include "llm/TransformerBloc.hpp"
#include "nn/Network.hpp"
#include "math/Matrix.hpp"
#include "math/Random.hpp"
#include "nn/SoftmaxLayer.hpp"

Network GPT(Index vocabSize, Index dimEmbedding, Index nbHead, Index nbTransformerBloc, Index contextMax, Random &gen);

#endif //RESEAUDENEURONEFROMSCRATCH_GPT_HPP
