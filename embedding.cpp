#include "embedding.h"

#include "common.h"


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
    float mem_size = static_cast<float > (this->d_model * this->vocab_size) * ggml_type_sizef(wtype);
    return static_cast<size_t>(mem_size);
}

void TokenEmbedding::init_params(struct ggml_context *ctx, ggml_type wtype) {
    this->word_embeddings = ggml_new_tensor_2d(ctx, wtype, this->d_model, this->vocab_size);
}

void TokenEmbedding::mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) {
    tensors[prefix + ".word_embeddings.weight"] = this->word_embeddings;
}


SinePositionalEmbedding::SinePositionalEmbedding(int64_t d_model, bool scale) {
    this->dim_model = d_model;
    if (scale) {
        this->x_scale = sqrtf(static_cast<float>(d_model));
    } else {
        this->x_scale = 1.0;
    }
    this->position_encodings = nullptr;
    this->alpha = nullptr;
//    this->dropout = nullptr;
//    this->reverse = false;
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

static void
ggml_vallex_set_pe(struct ggml_tensor *dst, const struct ggml_tensor *sin_t, const struct ggml_tensor *cos_t, int ith,
                   int nth, void *userdata) {
    GGML_ASSERT(userdata == nullptr);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(sin_t));
    GGML_ASSERT(ggml_is_contiguous(cos_t));

    float *sin_data = ggml_get_data_f32(sin_t);
    float *cos_data = ggml_get_data_f32(cos_t);
    float *dst_data = ggml_get_data_f32(dst);

    GGML_TENSOR_LOCALS(int64_t, sin_ne, sin_t, ne)
    GGML_TENSOR_LOCALS(size_t, sin_nb, sin_t, nb)

    GGML_TENSOR_LOCALS(int64_t, cos_ne, cos_t, ne)
    GGML_TENSOR_LOCALS(size_t, cos_nb, cos_t, nb)


    GGML_TENSOR_LOCALS(int64_t, dst_ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, dst_nb, dst, nb)

    const auto ne =  ggml_nelements(dst);
    const auto dr = (ne + nth - 1) / nth;
    const auto ie0 = dr * ith;
    const auto ie1 = std::min(ie0 + dr, ne);

    for (auto i = ie0; i < ie1; ++i) {
        /*const int row = i / dst_ne0; //row index*/
        const auto column = i % dst_ne0; //column index

        if (column == 0 || column % 2 == 0) {
            dst_data[i] = sin_data[column];
        } else {
            dst_data[i] = cos_data[column];
        }
    }
}

struct ggml_tensor *
SinePositionalEmbedding::extend_position_encodings(struct vallex_compute_context *ctx, ggml_tensor *x) {

    auto pe = ggml_new_tensor_2d(ctx->context, x->type, x->ne[1], this->dim_model);
    ggml_allocr_alloc(ctx->allocr, pe);
    if (!ggml_allocr_is_measure(ctx->allocr)) {
         ggml_set_f32(pe, 0.0);
    }

    auto position = ggml_new_tensor_2d(ctx->context, GGML_TYPE_F32, x->ne[1], 1);
    position = ggml_vallex_arange(ctx->context, position, 0, x->ne[1], 1);

    auto div_term = ggml_new_tensor_1d(ctx->context, GGML_TYPE_F32, this->dim_model);
    div_term = ggml_vallex_arange(ctx->context, div_term, 0, this->dim_model, 2);

    div_term = ggml_vallex_mul_num(
            ctx->context, div_term,
            -(logf(1000.0) / static_cast<float>(this->dim_model)));
    div_term = ggml_vallex_exp(ctx->context, div_term);

    auto sin_t = ggml_vallex_sin(ctx->context,
                                 ggml_mul(ctx->context, position, div_term));

    auto cos_t = ggml_vallex_cos(ctx->context,
                                 ggml_mul(ctx->context, position, div_term));

    pe = ggml_map_custom2(ctx->context,
                          sin_t,
                          cos_t,
                          ggml_vallex_set_pe,
                          GGML_N_TASKS_MAX,
                          nullptr);

    ggml_reshape_3d(ctx->context,
                    pe,
                    pe->ne[0],
                    pe->ne[1], 1);

    this->position_encodings = pe;

    return this->position_encodings;
}

struct ggml_tensor *SinePositionalEmbedding::forward(struct vallex_compute_context *ctx, ggml_tensor *x) {
    this->extend_position_encodings(ctx, x);
    //  output = x.unsqueeze(-1) if x.ndim == 2 else x
    //  output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
    auto output = x;
    if (x->n_dims == 2) {
        output->n_dims = 1;
    }
    const auto peShape = ggml_new_tensor(ctx->context, this->position_encodings->type, this->position_encodings->n_dims,
                                         this->position_encodings->ne);
    peShape->ne[1] = x->ne[1];

    output = ggml_add(
            ctx->context,
            ggml_vallex_mul_num(ctx->context, output, this->x_scale),
            ggml_mul(ctx->context,
                     this->alpha,
                     ggml_reshape(ctx->context, this->position_encodings, peShape))

    );


    return output;
}
