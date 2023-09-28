#include "embedding.h"

TokenEmbedding::TokenEmbedding(int d_model, int num_token, float dropout) {
    d_model = d_model;
    num_token = num_token;
    dropout = dropout;
}

void TokenEmbedding::set_data(ggml_tensor *view) {
    data = view;
}

ggml_tensor *TokenEmbedding::embedding() {

}