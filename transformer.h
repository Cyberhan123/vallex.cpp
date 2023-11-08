#pragma once
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "common.h"


class TransformerDecoderLayer
        : public NN::Module<
                /* tgt: Tensor,*/                                       ggml_tensor *,
                /* memory: Tensor*/                                     ggml_tensor *,
                /* tgt_mask: Optional[Tensor] = None, */                ggml_tensor *,
                /* memory_mask: Optional[Tensor] = None */              ggml_tensor *,
                /*tgt_key_padding_mask: Optional[Tensor] = None*/       ggml_tensor *,
                /*memory_key_padding_mask: Optional[Tensor] = None*/    ggml_tensor *
        > {

public:

    TransformerDecoderLayer(
            /* d_model: int,*/                                                  int64_t d_model,
            /* nhead: int,*/                                                    int64_t nhead,
            /*  dim_feedforward: int = 2048,*/                                  int dim_feedforward,
            /* dropout: float = 0.1,*/                                          float dropout,
            /* activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,*/   ggml_tensor *activation,
            /*linear1_self_attention_cls: nn.Module = nn.Linear,*/              ggml_tensor *linear1_self_attention_cls,
            /* linear2_self_attention_cls: nn.Module = nn.Linear,*/             ggml_tensor *linear2_self_attention_cls,
            /*linear1_feedforward_cls: nn.Module = nn.Linear,*/                 ggml_tensor *linear1_feedforward_cls,
            /*linear2_feedforward_cls: nn.Module = nn.Linear,*/                 ggml_tensor *linear2_feedforward_cls,
            /* batch_first: bool = False,*/                                     bool batch_first,
            /* norm_first: bool = False,*/                                      bool norm_first,
            /* device=None,*/
            /*dtype=None,*/                                                     ggml_type wtype,
            /*layer_norm_cls: nn.Module = LayerNorm,*/                          ggml_tensor *layer_norm_cls,
            /*layer_norm_eps: float = 1e-5,*/                                   float layer_norm_eps,
            /*adaptive_layer_norm=False,*/                                      ggml_tensor *adaptive_layer_norm
    );

    size_t compute_params_mem_size(ggml_type wtype) override;

    void init_params(struct ggml_context *ctx, ggml_type wtype) override;

    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override;

    struct ggml_tensor *
    forward(
            vallex_compute_context *ctx,
            /* tgt: Tensor,*/                                       ggml_tensor *tgt,
            /*memory: Tensor,*/                                     ggml_tensor *memory,
            /* tgt_mask: Optional[Tensor] = None,*/                 ggml_tensor *tgt_mask,
            /* memory_mask: Optional[Tensor] = None,*/              ggml_tensor *memory_mask,
            /*tgt_key_padding_mask: Optional[Tensor] = None,*/      ggml_tensor *tgt_key_padding_mask,
            /*memory_key_padding_mask: Optional[Tensor] = None,*/   ggml_tensor *memory_key_padding_mask
    ) override;

    bool norm_first;
private:
    struct ggml_tensor *self_attention_block(
            vallex_compute_context *ctx,
            ggml_tensor *x,
            ggml_tensor *attn_mask,
            ggml_tensor *key_padding_mask
    );

    struct ggml_tensor *multihead_attention_block(
            vallex_compute_context *ctx,
            ggml_tensor *x,
            ggml_tensor *mem,
            ggml_tensor *attn_mask,
            ggml_tensor *key_padding_mask
    );

    struct ggml_tensor * feed_forward_block(vallex_compute_context *ctx, ggml_tensor *x);
};


class TransformerEncoder :
        public NN::Module<
                /*src: Tensor,*/                                    ggml_tensor *,
                /*mask: Optional[Tensor] = None,*/                  ggml_tensor *,
                /*src_key_padding_mask: Optional[Tensor] = None,*/  ggml_tensor *,
                /*return_layer_states: bool = False,*/              bool
        > {
public:

    size_t compute_params_mem_size(ggml_type wtype) override;

    void init_params(struct ggml_context *ctx, ggml_type wtype) override;

    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override;

    struct ggml_tensor *
    forward(vallex_compute_context *ctx,
            /*src: Tensor,*/                                    ggml_tensor *src,
            /*mask: Optional[Tensor] = None,*/                  ggml_tensor *mask,
            /*src_key_padding_mask: Optional[Tensor] = None,*/  ggml_tensor *src_key_padding_mask,
            /*return_layer_states: bool = False,*/              bool return_layer_states
    ) override;
};

class TransformerEncoderLayer {

};

#endif