#pragma once
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <cmath>
#include "common.h"
#include <ggml/ggml.h>


class TokenEmbedding : public NN::Module<ggml_tensor *> {
public:
    explicit TokenEmbedding(int64_t d_model, int64_t vocab_size);

    size_t compute_params_mem_size(ggml_type wtype) override;

    void init_params(struct ggml_context *ctx, ggml_type wtype) override;

    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override;

    struct ggml_tensor *forward(struct vallex_compute_context *ctx, ggml_tensor *x) override;

protected:
    int64_t d_model;
    int64_t vocab_size;
    struct ggml_tensor *word_embeddings;
};


class SinePositionalEmbedding : public NN::Module<ggml_tensor *> {
public:
    SinePositionalEmbedding(
            int64_t d_model,
            bool scale
    );

    size_t compute_params_mem_size(ggml_type wtype) override;

    void init_params(struct ggml_context *ctx, ggml_type wtype) override;

    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override;

    struct ggml_tensor *forward(struct vallex_compute_context *ctx, ggml_tensor *x) override;

    struct ggml_tensor * extend_position_encodings(struct vallex_compute_context *ctx, ggml_tensor *x);

protected:


private:
    int64_t dim_model;
    float x_scale;
    ggml_tensor *alpha;
    void *dropout;
    bool reverse;
    ggml_tensor *position_encodings;
};


#endif