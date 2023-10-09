#include "embedding.h"

SinePositionalEmbedding::SinePositionalEmbedding(
        int d_model,
        float dropout,
        bool scale,
        bool normalize
) {
    this->dim_model = dim_model;
    if (scale) {
        this->x_scale = std::sqrt(dim_model);
    } else {
        this->x_scale = 1.0;
    }
}

TokenEmbedding::TokenEmbedding(int d_model, int num_token, float dropout) {
    this->d_model = d_model;
    this->num_token = num_token;
    this->dropout = dropout;
}

void TokenEmbedding::set_data(ggml_tensor *view) {
    data = view;
}

ggml_tensor *TokenEmbedding::forward(ggml_tensor *x) {
    return nullptr;
}
