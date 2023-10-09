#pragma once
#ifndef VALLEX_H
#define VALLEX_H

#include <string>
#include <map>

#include "embedding.h"
#include "transformer.h"
#include "ggml/ggml.h"

class VALLF {
public:
    VALLF(
            int d_model,
            int nhead,
            int num_layers,
            bool norm_first = true,
            bool add_prenet = false,
            TransformerDecoder *decoder_cls = nullptr,
            TransformerDecoderLayer *decoder_layer_cls = nullptr,
            int prefix_mode = 0,
            bool share_embedding = true,
            float nar_scale_factor = 1.0,
            bool prepend_bos = true,
            int num_token = 1024,
            int num_quantizers = 8
    );

    bool load_model_from_file();


private:
    int nar_d_model;
    TokenEmbedding *ar_text_embedding;
    TokenEmbedding *nar_text_embedding;
    bool ar_audio_prepend_bos;

    void *ar_text_prenet;
    void *nar_audio_prenet;

    SinePositionalEmbedding *ar_text_position;
    SinePositionalEmbedding *ar_audio_position;

    TransformerDecoder *ar_decoder;
    TransformerDecoderLayer *ar_predict_layer;
    int num_heads;
    int prefix_mode{};
    void *nar_audio_embeddings;
    SinePositionalEmbedding *nar_text_position;
    void *nar_decoder;
    void *nar_predict_layers;
    void *nar_stage_embeddings;
};


class VALLE : public VALLF {
public:
    VALLE(int d_model,
          int nhead,
          int num_layers,
          bool norm_first,
          bool add_prenet,
          int prefix_mode,
          bool share_embedding,
          float nar_scale_factor,
          bool prepend_bos,
          int num_token,
          int quantizers);

    ggml_tensor *inference(
            ggml_tensor *x,
            ggml_tensor *x_lens,
            ggml_tensor *y,
            ggml_tensor *enroll_x_lens,
            int top_k,
            float temperature,
            const std::string& prompt_language,
            const std::string& text_language,
            int best_of,
            float length_penalty,
            bool return_worst = false
    );

private:
    const std::map<std::string, int> language_ID = std::map<std::string, int>{
            {"en", 0},
            {"zh", 1},
            {"ja", 2}
    };
    TokenEmbedding *ar_language_embedding;
    TokenEmbedding *nar_language_embedding;
};

#endif