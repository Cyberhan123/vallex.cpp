#include "embedding.h"

#include "common.h"
#include "ggml/ggml.h"


#include <vector>

TokenEmbedding::TokenEmbedding(int64_t d_model, int64_t vocab_size) {
    this->d_model = d_model;
    this->vocab_size = vocab_size;
    this->word_embeddings = nullptr;
}


struct ggml_tensor *TokenEmbedding::forward(struct vallex_compute_context *ctx, ggml_tensor *x) {
    const auto emb = ggml_get_rows(ctx->context, word_embeddings, x);
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


SinePositionalEmbedding::SinePositionalEmbedding(int64_t d_model, bool scale) {
    this->dim_model = d_model;
    if (scale) {
        this->x_scale = std::sqrt(d_model);
    } else {
        this->x_scale = 1.0;
    }
    this->position_encodings = nullptr;

}

size_t SinePositionalEmbedding::compute_params_mem_size(ggml_type wtype) {
    float mem_size = static_cast<float>(this->dim_model) * ggml_type_sizef(wtype);
    return static_cast<size_t>(mem_size);
}

void SinePositionalEmbedding::init_params(struct ggml_context *ctx, ggml_type wtype) {
    this->alpha = ggml_new_tensor_1d(ctx, wtype, 1);
}

void SinePositionalEmbedding::mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) {
    tensors[prefix + ".alpha"] = alpha;
}

struct ggml_tensor * SinePositionalEmbedding::extend_position_encodings(struct vallex_compute_context *ctx, ggml_tensor *x)  {
    auto pe = ggml_new_tensor_2d(ctx->context, x->type, x->ne[1], this->dim_model);
    ggml_allocr_alloc(ctx->allocr, pe);
    if (!ggml_allocr_is_measure(ctx->allocr)) {
        pe = ggml_set_f32(pe, 0.0);
    }

    const auto position = ggml_vallex_arange_2d(ctx, 0, x->ne[1], 1);

    auto div_term = ggml_vallex_arange_1d(ctx, 0, this->dim_model, 2);

    const auto p_data = logf(1000.0) / static_cast<float>(this->dim_model);

    auto p = ggml_new_tensor_1d(ctx->context, GGML_TYPE_F32, this->dim_model);
    ggml_allocr_alloc(ctx->allocr, p);
    if (!ggml_allocr_is_measure(ctx->allocr)) {
        ggml_set_f32(p, p_data);
    }

    div_term = ggml_mul(ctx->context, div_term, p);
    div_term = ggml_vallex_exp(ctx->context, div_term);
//    auto sin_t = ggml_vallex_sin(ctx->context, ggml_mul(ctx->context, position, div_term));
//    auto cos_t = ggml_vallex_cos(ctx->context, ggml_mul(ctx->context, position, div_term));
    return div_term;


//
//    div_term = mult_1d(div_term, -p);
//    auto c = sin_1d(mult_1d(position, div_term));
//
//    for (auto row = 0; row < pe.size(); row++) {
//        for (auto col = 0; col < pe[row].size(); col += 2) {
//
//            pe[row][col] = c.data();
//        }
//    }

}

struct ggml_tensor *SinePositionalEmbedding::forward(struct vallex_compute_context *ctx, ggml_tensor *x) {
    return nullptr;
}
