#include "llm/GPT.hpp"

Network GPT(Index vocabSize, Index dimEmbedding, Index nbHead, Index nbTransformerBloc, Index contextMax, Random &gen) {
    Network net;
    net.add(std::make_unique<EmbeddingLayer>(vocabSize, dimEmbedding, contextMax, gen));
    for (Index n =0; n < nbTransformerBloc; n++) {
        net.add(std::make_unique<TransformerBloc>(dimEmbedding,nbHead,gen));
    }
    net.add(std::make_unique<NormLayer>(dimEmbedding));
    net.add(std::make_unique<DenseLayer>(dimEmbedding,vocabSize,gen,2));
    net.add(std::make_unique<SoftmaxLayer>(vocabSize));
    net.set_actual_context(contextMax);
    return net;
}
