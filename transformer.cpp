#include "transformer.h"

TransformerDecoderLayer::TransformerDecoderLayer(
        int64_t d_model,
        int64_t nhead,
        int dim_feedforward,
        float dropout,
        ggml_tensor *activation,
        ggml_tensor *linear1_self_attention_cls,
        ggml_tensor *linear2_self_attention_cls,
        ggml_tensor *linear1_feedforward_cls,
        ggml_tensor *linear2_feedforward_cls,
        bool batch_first,
        bool norm_first, ggml_type wtype,
        ggml_tensor *layer_norm_cls,
        float layer_norm_eps,
        ggml_tensor *adaptive_layer_norm
) {

}


size_t TransformerDecoderLayer::compute_params_mem_size(ggml_type wtype) {
    return 0;
}

void TransformerDecoderLayer::init_params(struct ggml_context *ctx, ggml_type wtype) {

}

void TransformerDecoderLayer::mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) {

}

struct ggml_tensor *TransformerDecoderLayer::forward(vallex_compute_context *ctx, ggml_tensor *tgt, ggml_tensor *memory,
                                                     ggml_tensor *tgt_mask, ggml_tensor *memory_mask,
                                                     ggml_tensor *tgt_key_padding_mask,
                                                     ggml_tensor *memory_key_padding_mask) {
    return nullptr;
}


size_t TransformerEncoder::compute_params_mem_size(ggml_type wtype) {
    return 0;
}

void TransformerEncoder::init_params(struct ggml_context *ctx, ggml_type wtype) {

}

void TransformerEncoder::mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) {

}

struct ggml_tensor *TransformerEncoder::forward(vallex_compute_context *ctx, ggml_tensor *src, ggml_tensor *mask,
                                                ggml_tensor *src_key_padding_mask, bool return_layer_states) {
    return nullptr;
}
