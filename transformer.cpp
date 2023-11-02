#include "transformer.h"

struct ggml_tensor *
forward(vallex_compute_context *ctx,
        ggml_tensor *tgt,
        ggml_tensor *memory,
        ggml_tensor *tgt_mask,
        ggml_tensor *memory_mask,
        ggml_tensor *tgt_key_padding_mask,
        ggml_tensor *memory_key_padding_mask
) {
    return nullptr;
}