#include "transformer.h"
#include "activation.h"

TransformerEncoderLayer::TransformerEncoderLayer(
    int64_t d_model,
    int64_t nhead,
    int dim_feedforward,
    float dropout,
    ggml_tensor* activation,
    ggml_tensor* linear1_self_attention_cls,
    ggml_tensor* linear2_self_attention_cls,
    ggml_tensor* linear1_feedforward_cls,
    ggml_tensor* linear2_feedforward_cls,
    bool batch_first,
    bool norm_first,
    ggml_type wtype,
    ggml_tensor* layer_norm_cls,
    float layer_norm_eps,
    ggml_tensor* adaptive_layer_norm)
{
    this->d_model = d_model;
    this->nhead = nhead;
    this->self_attn = new MultiheadAttention(
        d_model,
        nhead,
        dropout,
        true,
        false,
        false,
        0,
        0,
        batch_first,
        linear1_self_attention_cls,
        linear2_self_attention_cls,
        wtype
    );
    this->layer_norm_eps = layer_norm_eps;
}

size_t TransformerEncoderLayer::compute_params_mem_size(ggml_type wtype)
{
    float mem_size = static_cast<float>(this->nhead * this->d_model) * ggml_type_sizef(wtype);
    return static_cast<size_t>(mem_size);
}

void TransformerEncoderLayer::init_params(struct ggml_context* ctx, ggml_type wtype)
{
    this->linear1_weight = ggml_new_tensor_2d(ctx, wtype, this->nhead, this->d_model);
    this->linear1_bias = ggml_new_tensor_1d(ctx, wtype, this->nhead);
    this->linear2_weight = ggml_new_tensor_2d(ctx, wtype, this->nhead, this->d_model);
    this->linear2_bias = ggml_new_tensor_1d(ctx, wtype, this->nhead);
}

void TransformerEncoderLayer::mapping_tensor(std::map<std::string, struct ggml_tensor*>& tensors, std::string prefix)
{
    tensors[prefix + ".linear1.weight"] = this->linear1_weight;
    tensors[prefix + ".linear1.bias"] = this->linear1_bias;
    tensors[prefix + ".linear2.weight"] = this->linear2_weight;
    tensors[prefix + ".linear2.bias"] = this->linear2_bias;
}

struct ggml_tensor* TransformerEncoderLayer::forward(
    vallex_compute_context* ctx,
    ggml_tensor* src,
    ggml_tensor* src_mask,
    ggml_tensor* src_key_padding_mask)
{
    // const auto n_element = ggml_nelements(src);
    // auto is_src_tuple = false;
    ggml_tensor* x;
    // ggml_tensor* stage_embedding;
    // if (n_element == 2)
    // {
    //     x = ggml_view_1d(ctx->context, src, 1, 0);
    //     stage_embedding = ggml_view_1d(ctx->context, src, 1, sizeof(float));
    //     is_src_tuple = true;
    // }
    // else
    // {
    //     x = ggml_view_tensor(ctx->context, src);
    // }

    //    if self.norm_first:
    //    x = x + self._sa_block(
    //            self.norm1(x, stage_embedding),
    //            src_mask,
    //            src_key_padding_mask,
    //    )
    //    x = x + self._ff_block(self.norm2(x, stage_embedding))
    //    else:
    //    x = self.norm1(
    //            x + self._sa_block(x, src_mask, src_key_padding_mask),
    //            stage_embedding,
    //    )
    //    x = self.norm2(x + self._ff_block(x), stage_embedding)
    //
    //    if is_src_tuple:
    //        return (x, stage_embedding)
    //    return x
    if (this->norm_first)
    {
        x = ggml_add(ctx->context, x, this->self_attention_block(
                         ctx,
                         ggml_norm(ctx->context, x, this->layer_norm_eps),
                         src_mask,
                         src_key_padding_mask
                         //
                     ));
        x = ggml_add(ctx->context, x, this->feed_forward_block(ctx, ggml_norm(ctx->context, x, this->layer_norm_eps)));
    }
    else
    {
        x = ggml_norm(ctx->context,
                      ggml_add(ctx->context, x,
                               this->self_attention_block(ctx, x, src_mask, src_key_padding_mask))
                      this->layer_norm_eps
        );
        x = ggml_norm(ctx->context,
                      ggml_add(ctx->context,
                               x,
                               this->feed_forward_block(ctx, x)
                      ),
                      this->layer_norm_eps
        );
    }
    return x;
}

struct ggml_tensor*
TransformerEncoderLayer::self_attention_block(vallex_compute_context* ctx, ggml_tensor* x, ggml_tensor* attn_mask,
                                              ggml_tensor* key_padding_mask)
{
    // this->self_attn->forward(ctx,x,x,x,attn_mask,key_padding_mask,false,false);
    return nullptr;
}

struct ggml_tensor*
TransformerEncoderLayer::multihead_attention_block(vallex_compute_context* ctx, ggml_tensor* x, ggml_tensor* mem,
                                                   ggml_tensor* attn_mask, ggml_tensor* key_padding_mask)
{
    return nullptr;
}

struct ggml_tensor* TransformerEncoderLayer::feed_forward_block(vallex_compute_context* ctx, ggml_tensor* x)
{
    return nullptr;
}


TransformerEncoder::TransformerEncoder(const TransformerEncoderLayer& encoder_layer, int64_t num_layers)
{
    for (int i = 0; i < num_layers; ++i)
    {
        this->layers.push_back(encoder_layer);
    }

    this->num_layers = num_layers;
    //    this->norm1.identityNorm = norm;
}

size_t TransformerEncoder::compute_params_mem_size(ggml_type wtype)
{
    return 0;
}

void TransformerEncoder::init_params(struct ggml_context* ctx, ggml_type wtype)
{
}

void TransformerEncoder::mapping_tensor(std::map<std::string, struct ggml_tensor*>& tensors, std::string prefix)
{
}

struct ggml_tensor* TransformerEncoder::forward(vallex_compute_context* ctx, ggml_tensor* src, ggml_tensor* mask,
                                                ggml_tensor* src_key_padding_mask, bool return_layer_states)
{
    return nullptr;
}

struct ggml_tensor* TransformerEncoder::infer(
    vallex_compute_context* ctx,
    ggml_tensor* src,
    ggml_tensor* mask,
    ggml_tensor* src_key_padding_mask,
    bool return_layer_states,
    ggml_tensor* past_kv,
    bool use_cache
)
{
    return nullptr;
}
