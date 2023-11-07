#ifndef VALLEX_CPP_ACTIVATION_H
#define VALLEX_CPP_ACTIVATION_H


#include "common.h"

class MultiheadAttention : public NN::Module<
        /* x: Tensor,*/                                 ggml_tensor *,
        /*key_padding_mask: Optional[Tensor] = None,*/  ggml_tensor *,
        /*need_weights: bool = True,*/                  bool,
        /*attn_mask: Optional[Tensor] = None,*/         ggml_tensor *,
        /*average_attn_weights: bool = True,*/          bool,
        /*past_kv = None,*/                             ggml_tensor *,
        /*use_cache = False*/                           bool
> {

public:
    MultiheadAttention(
            int64_t embed_dim,
            int64_t num_heads,
            float dropout,
            bool bias,
            bool add_bias_kv,
            bool add_zero_attn,
            int64_t kdim,
            int64_t vdim,
            bool batch_first,
            ggml_tensor *linear1_cls,
            ggml_tensor *linear2_cls,
            /*ggml_tensor*   device=,*/
            ggml_type dtype
    );

public:
    size_t compute_params_mem_size(ggml_type wtype) override;

    void init_params(struct ggml_context *ctx, ggml_type wtype) override;

    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override;

    struct ggml_tensor *
    forward(
            vallex_compute_context *ctx,
            /* x: Tensor,*/                                 ggml_tensor *x,
            /*key_padding_mask: Optional[Tensor] = None,*/  ggml_tensor *key_padding_mask,
            /*need_weights: bool = True,*/                  bool need_weights,
            /*attn_mask: Optional[Tensor] = None,*/         ggml_tensor *attn_mask,
            /*average_attn_weights: bool = True,*/          bool average_attn_weights,
            /*past_kv = None,*/                             ggml_tensor *past_kv,
            /*use_cache = False*/                           bool use_cache
    ) override;

    struct ggml_tensor *in_proj_weight;
    struct ggml_tensor *in_proj_bias;
    struct ggml_tensor *out_proj_weight;
    struct ggml_tensor *out_proj_bias;
    int64_t num_heads;
};


#endif //VALLEX_CPP_ACTIVATION_H
