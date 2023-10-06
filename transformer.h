#pragma once
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

class TransformerDecoder {
   int TokenEmbeddingImpl(int dim_model, int vocab_size, float dropout = 0.0);
};

class TransformerDecoderLayer {
};
#endif