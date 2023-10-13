#include "embedding.h"

#include "ggml/ggml.h"
#include "common.h"

#include <vector>

TokenEmbedding::TokenEmbedding(int32_t d_model, size_t vocab_size) {
    this->d_model = d_model;
    this->vocab_size = vocab_size;
}


struct ggml_tensor *TokenEmbedding::forward(struct ggml_graph_ctx *ctx, ggml_tensor *x) {
    const auto emb = ggml_get_rows(ctx->ctx, word_embeddings, x);
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


SinePositionalEmbedding::SinePositionalEmbedding(int d_model, bool scale) {
    this->dim_model = d_model;
    if (scale) {
        this->x_scale = std::sqrt(d_model);
    } else {
        this->x_scale = 1.0;
    }
}

size_t SinePositionalEmbedding::compute_params_mem_size(ggml_type wtype) {
    double mem_size = this->dim_model * ggml_type_sizef(wtype);
    return static_cast<size_t>(mem_size);
}

void SinePositionalEmbedding::init_params(struct ggml_context *ctx, ggml_type wtype) {
    this->alpha = ggml_new_tensor_1d(ctx, wtype, 1);
}

void SinePositionalEmbedding::mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) {
    tensors[prefix + ".alpha"] = alpha;
}

void SinePositionalEmbedding::extend_position_encodings(struct ggml_graph_ctx *ctx, ggml_tensor *x) {
    this->position_encodings = ggml_new_tensor_2d(ctx->ctx, x->type, x->ne[1], this->dim_model);
    this->position_encodings = ggml_set_zero(this->position_encodings);
    const auto position = ggml_arange_2d(ctx, GGML_TYPE_F32, 0, x->ne[1], 1);
    auto div_term = ggml_arange_1d(ctx, GGML_TYPE_F32, 0, this->dim_model, 2);
    const auto p = static_cast<float > (std::log(1000.0) / this->dim_model);
    const auto pt = ggml_new_tensor_1d(ctx->ctx, GGML_TYPE_F32, 1);
    if (!ggml_allocr_is_measure(ctx->allocr)) {
        ggml_backend_tensor_set(pt, &p, 0,  sizeof(float));
    }
    div_term = ggml_mul(ctx->ctx, div_term, pt)

}

struct ggml_tensor *SinePositionalEmbedding::forward(struct ggml_graph_ctx *ctx, ggml_tensor *x) {
    return nullptr;
}
