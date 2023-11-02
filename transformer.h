#pragma once
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "common.h"

class TransformerDecoder {

};


class TransformerDecoderLayer
        : public NN::Module<
                /* tgt: Tensor,*/                                       ggml_tensor *,
                /* memory: Tensor*/                                     ggml_tensor *,
                /* tgt_mask: Optional[Tensor] = None, */                ggml_tensor *,
                /* memory_mask: Optional[Tensor] = None */              ggml_tensor *,
                /*tgt_key_padding_mask: Optional[Tensor] = None*/       ggml_tensor *,
                /*memory_key_padding_mask: Optional[Tensor] = None*/    ggml_tensor *
        > {

    struct ggml_tensor *
    forward(vallex_compute_context *ctx,
            ggml_tensor *tgt,
            ggml_tensor *memory,
            ggml_tensor *tgt_mask,
            ggml_tensor *memory_mask,
            ggml_tensor *tgt_key_padding_mask,
            ggml_tensor *memory_key_padding_mask
            ) override;
};

class TransformerEncoder {

};

class TransformerEncoderLayer {

};

#endif