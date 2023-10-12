#include "embedding.h"

TokenEmbedding::TokenEmbedding(int32_t d_model, size_t vocab_size) {
    this->d_model = d_model;
    this->vocab_size = vocab_size;
}


struct ggml_tensor *TokenEmbedding::forward(struct ggml_context *ctx, ggml_tensor *x) {
    const auto emb = ggml_get_rows(ctx, word_embeddings, x);
    return emb;
}

size_t TokenEmbedding::compute_params_mem_size(ggml_type wtype) {
    double mem_size = this->d_model * this->vocab_size * ggml_type_sizef(wtype);
    return static_cast<size_t>(mem_size);
}

void TokenEmbedding::init_params(struct ggml_context *ctx, ggml_type wtype) {
    this->word_embeddings = ggml_new_tensor_2d(ctx, wtype, this->d_model, this->vocab_size);
}

void TokenEmbedding::mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) {
    tensors[prefix + ".word_embeddings.weight"] = word_embeddings;
}


SinePositionalEmbedding::SinePositionalEmbedding(int d_model, bool scale, bool alpha) {
    this->dim_model = d_model;
    if (scale){
        this->x_scale=std::sqrt(d_model);
    }else{
        this->x_scale = 1.0;
    }


}

size_t SinePositionalEmbedding::compute_params_mem_size(ggml_type wtype) {
    return 0;
}

void SinePositionalEmbedding::init_params(struct ggml_context *ctx, ggml_type wtype) {
    this->alpha = ggml_new_tensor_1d(ctx, wtype, 1);
}

void SinePositionalEmbedding::mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) {

}

struct ggml_tensor *SinePositionalEmbedding::forward(struct ggml_context *ctx, ggml_tensor *x) {
    return nullptr;
}
