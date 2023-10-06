#pragma once
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "ggml.h"

class SinePositionalEmbedding {
public:
    SinePositionalEmbedding(int d_model, float dropout, bool scale, bool normalize);
};

class TokenEmbedding {
public:
    int d_model;
    int num_token;
    float dropout;
    ggml_tensor *data;

    TokenEmbedding(int d_model, int num_token, float dropout = 0.0);

    void set_data(ggml_tensor *data);

    ggml_tensor *embedding();
};

#endif