#pragma once
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "common.h"
#include "activation.h"
#include "vector"


//class LayerNorm : public NN::Module<ggml_tensor *, ggml_tensor *> {
//public:
//    LayerNorm(
//            struct ggml_tensor *normalized_shape,
//            float eps,
//            bool elementwise_affine
//    ) {
//        this->eps = eps;
//        this->elementwise_affine = elementwise_affine;
//        this->normalized_shape = normalized_shape;
////        if self.elementwise_affine:
////        self.weight = nn.Parameter(
////                torch.empty(self.normalized_shape, **factory_kwargs)
////        )
////        self.bias = nn.Parameter(
////                torch.empty(self.normalized_shape, **factory_kwargs)
////        )
////        else:
////        self.register_parameter("weight", None)
////        self.register_parameter("bias", None)
//
//    };
//
//    size_t compute_params_mem_size(ggml_type wtype) override {
//        return 0;
//    };
//
//    void init_params(struct ggml_context *ctx, ggml_type wtype) override {
//
//    };
//
//    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override {};
//
//    struct ggml_tensor *forward(vallex_compute_context *ctx, ggml_tensor *input, ggml_tensor *embedding) override {
//        if (ggml_nelements(input) == 2) {
//            auto x = ggml_view_1d(ctx->context, input, 1, 0);
//            embedding = ggml_view_1d(ctx->context, input, 1, sizeof(float));
//            return ggml_vallex_layer_norm(ctx->context, x, this->normalized_shape, this->weight, this->bias, this->eps);
//        }
//        GGML_ASSERT(embedding == nullptr);
//        return ggml_vallex_layer_norm(ctx->context, input, this->normalized_shape, this->weight, this->bias, this->eps);
//    };
//    struct ggml_tensor *normalized_shape;
//    float eps;
//    bool elementwise_affine;
//    struct ggml_tensor *weight;
//    struct ggml_tensor *bias;
//};
//
//class AdaptiveLayerNorm : public NN::Module<ggml_tensor *, ggml_tensor *> {
//public:
//    AdaptiveLayerNorm(int64_t d_model, NN::Module<struct ggml_tensor *, struct ggml_tensor *> norm) {
//
//    };
//
//    size_t compute_params_mem_size(ggml_type wtype) override {
//        return 0;
//    };
//
//    void init_params(struct ggml_context *ctx, ggml_type wtype) override {
//
//    };
//
//    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override {};
//
//    struct ggml_tensor *forward(vallex_compute_context *ctx, ggml_tensor *input, ggml_tensor *embedding) override {
//
//        return nullptr;
//    };
//
//};
//
//class BasicNorm : public NN::Module<ggml_tensor *, ggml_tensor *> {
//    size_t compute_params_mem_size(ggml_type wtype) override {
//        return 0;
//    };
//
//    void init_params(struct ggml_context *ctx, ggml_type wtype) override {
//
//    };
//
//    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override {};
//
//    struct ggml_tensor *forward(vallex_compute_context *ctx, ggml_tensor *input, ggml_tensor *embedding) override {
//
//        return nullptr;
//    };
//};

//class BalancedBasicNorm : public NN::Module<ggml_tensor *, ggml_tensor *> {
//public:
//    BalancedBasicNorm() {};
//
//    size_t compute_params_mem_size(ggml_type wtype) override {
//        return 0;
//    };
//
//    void init_params(struct ggml_context *ctx, ggml_type wtype) override {
//
//    };
//
//    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override {};
//
//    struct ggml_tensor *forward(vallex_compute_context *ctx, ggml_tensor *input, ggml_tensor *embedding) override {
//
//        return nullptr;
//    };
//};

class IdentityNorm : public NN::Module<ggml_tensor *, ggml_tensor *, ggml_tensor *> {
public:
    IdentityNorm() {};

    size_t compute_params_mem_size(ggml_type wtype) override {
        return 0;
    };

    void init_params(struct ggml_context *ctx, ggml_type wtype) override {

    };

    void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) override {};

    struct ggml_tensor *forward(vallex_compute_context *ctx, ggml_tensor *input, ggml_tensor *embedding) override {

        return nullptr;
    };
};

//union Norm {
//    LayerNorm layerNorm;
//    AdaptiveLayerNorm adaptiveLayerNorm;
//    BalancedBasicNorm balancedBasicNorm;
//    IdentityNorm identityNorm;
//};

class TransformerEncoderLayer
        : public NN::Module<
                ggml_tensor *,
                /*src: Tensor,*/                                    ggml_tensor *,
                /*src_mask: Optional[Tensor] = None,*/              ggml_tensor *,
                /*src_key_padding_mask: Optional[Tensor] = None,*/  ggml_tensor *
        > {

public:

    TransformerEncoderLayer(
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
            /* tgt: Tensor,*/                                       ggml_tensor *src,
            /*memory: Tensor,*/                                     ggml_tensor *src_mask,
            /* tgt_mask: Optional[Tensor] = None,*/                 ggml_tensor *src_key_padding_mask
    ) override;

    bool norm_first;

    MultiheadAttention *self_attn;

    int64_t d_model;

    int64_t nhead;

    ggml_tensor *linear1_weight;
    ggml_tensor *linear2_weight;
    ggml_tensor *linear1_bias;
    ggml_tensor *linear2_bias;

    float layer_norm_eps;
private:
    struct ggml_tensor *self_attention_block(
            vallex_compute_context *ctx,
            ggml_tensor *x,
            ggml_tensor *attn_mask,
            ggml_tensor *key_padding_mask
    );

//    struct ggml_tensor *multihead_attention_block(
//            vallex_compute_context *ctx,
//            ggml_tensor *x,
//            ggml_tensor *mem,
//            ggml_tensor *attn_mask,
//            ggml_tensor *key_padding_mask
//    );

    struct ggml_tensor *feed_forward_block(vallex_compute_context *ctx, ggml_tensor *x);
};


class TransformerEncoder :
        public NN::Module<
                ggml_tensor *,
                /*src: Tensor,*/                                    ggml_tensor *,
                /*mask: Optional[Tensor] = None,*/                  ggml_tensor *,
                /*src_key_padding_mask: Optional[Tensor] = None,*/  ggml_tensor *,
                /*return_layer_states: bool = False,*/              bool
        > {
public:
    TransformerEncoder(const TransformerEncoderLayer &encoder_layer, int64_t num_layers);

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

    struct ggml_tensor *
    infer(
            vallex_compute_context *ctx,
            /*src: Tensor,*/                                    ggml_tensor *src,
            /* mask: Optional[Tensor] = None,*/                 ggml_tensor *mask,
            /*src_key_padding_mask: Optional[Tensor] = None,*/  ggml_tensor *src_key_padding_mask,
            /*return_layer_states: bool = False,*/              bool return_layer_states,
            /*past_kv: Optional[Tensor] = None,*/               ggml_tensor *past_kv,
            /*use_cache: bool = False,*/                        bool use_cache
    );

    std::vector<TransformerEncoderLayer> layers;
    int64_t num_layers;
//    Norm norm1;
};

#endif