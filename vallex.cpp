#include <string>

#include "vallex.h"
#include "embedding.h"
#include "ggml.h"



class TransformerDecoder {
    TokenEmbeddingImpl(int dim_model, int vocab_size, float dropout = 0.0)
};

class TransformerDecoderLayer {
};

const auto mapKeys =[
        'ar_text_embedding.word_embeddings.weight',
        'nar_text_embedding.word_embeddings.weight',
        'ar_audio_embedding.word_embeddings.weight',
        'ar_text_position.alpha',
        'ar_audio_position.alpha',
        'ar_decoder.layers.0.self_attn.in_proj_weight',
        'ar_decoder.layers.0.self_attn.in_proj_bias',
        'ar_decoder.layers.0.self_attn.out_proj.weight',
        'ar_decoder.layers.0.self_attn.out_proj.bias',
        'ar_decoder.layers.0.linear1.weight',
        'ar_decoder.layers.0.linear1.bias',
        'ar_decoder.layers.0.linear2.weight',
        'ar_decoder.layers.0.linear2.bias',
        'ar_decoder.layers.0.norm1.weight',
        'ar_decoder.layers.0.norm1.bias',
        'ar_decoder.layers.0.norm2.weight',
        'ar_decoder.layers.0.norm2.bias',
        'ar_decoder.layers.1.self_attn.in_proj_weight',
        'ar_decoder.layers.1.self_attn.in_proj_bias',
        'ar_decoder.layers.1.self_attn.out_proj.weight',
        'ar_decoder.layers.1.self_attn.out_proj.bias',
        'ar_decoder.layers.1.linear1.weight',
        'ar_decoder.layers.1.linear1.bias',
        'ar_decoder.layers.1.linear2.weight',
        'ar_decoder.layers.1.linear2.bias',
        'ar_decoder.layers.1.norm1.weight',
        'ar_decoder.layers.1.norm1.bias',
        'ar_decoder.layers.1.norm2.weight',
        'ar_decoder.layers.1.norm2.bias',
        'ar_decoder.layers.2.self_attn.in_proj_weight',
        'ar_decoder.layers.2.self_attn.in_proj_bias',
        'ar_decoder.layers.2.self_attn.out_proj.weight',
        'ar_decoder.layers.2.self_attn.out_proj.bias',
        'ar_decoder.layers.2.linear1.weight',
        'ar_decoder.layers.2.linear1.bias',
        'ar_decoder.layers.2.linear2.weight',
        'ar_decoder.layers.2.linear2.bias',
        'ar_decoder.layers.2.norm1.weight',
        'ar_decoder.layers.2.norm1.bias',
        'ar_decoder.layers.2.norm2.weight',
        'ar_decoder.layers.2.norm2.bias',
        'ar_decoder.layers.3.self_attn.in_proj_weight',
        'ar_decoder.layers.3.self_attn.in_proj_bias',
        'ar_decoder.layers.3.self_attn.out_proj.weight',
        'ar_decoder.layers.3.self_attn.out_proj.bias',
        'ar_decoder.layers.3.linear1.weight',
        'ar_decoder.layers.3.linear1.bias',
        'ar_decoder.layers.3.linear2.weight',
        'ar_decoder.layers.3.linear2.bias',
        'ar_decoder.layers.3.norm1.weight',
        'ar_decoder.layers.3.norm1.bias',
        'ar_decoder.layers.3.norm2.weight',
        'ar_decoder.layers.3.norm2.bias',
        'ar_decoder.layers.4.self_attn.in_proj_weight',
        'ar_decoder.layers.4.self_attn.in_proj_bias',
        'ar_decoder.layers.4.self_attn.out_proj.weight',
        'ar_decoder.layers.4.self_attn.out_proj.bias',
        'ar_decoder.layers.4.linear1.weight',
        'ar_decoder.layers.4.linear1.bias',
        'ar_decoder.layers.4.linear2.weight',
        'ar_decoder.layers.4.linear2.bias',
        'ar_decoder.layers.4.norm1.weight',
        'ar_decoder.layers.4.norm1.bias',
        'ar_decoder.layers.4.norm2.weight',
        'ar_decoder.layers.4.norm2.bias',
        'ar_decoder.layers.5.self_attn.in_proj_weight',
        'ar_decoder.layers.5.self_attn.in_proj_bias',
        'ar_decoder.layers.5.self_attn.out_proj.weight',
        'ar_decoder.layers.5.self_attn.out_proj.bias',
        'ar_decoder.layers.5.linear1.weight',
        'ar_decoder.layers.5.linear1.bias',
        'ar_decoder.layers.5.linear2.weight',
        'ar_decoder.layers.5.linear2.bias',
        'ar_decoder.layers.5.norm1.weight',
        'ar_decoder.layers.5.norm1.bias',
        'ar_decoder.layers.5.norm2.weight',
        'ar_decoder.layers.5.norm2.bias',
        'ar_decoder.layers.6.self_attn.in_proj_weight',
        'ar_decoder.layers.6.self_attn.in_proj_bias',
        'ar_decoder.layers.6.self_attn.out_proj.weight',
        'ar_decoder.layers.6.self_attn.out_proj.bias',
        'ar_decoder.layers.6.linear1.weight',
        'ar_decoder.layers.6.linear1.bias',
        'ar_decoder.layers.6.linear2.weight',
        'ar_decoder.layers.6.linear2.bias',
        'ar_decoder.layers.6.norm1.weight',
        'ar_decoder.layers.6.norm1.bias',
        'ar_decoder.layers.6.norm2.weight',
        'ar_decoder.layers.6.norm2.bias',
        'ar_decoder.layers.7.self_attn.in_proj_weight',
        'ar_decoder.layers.7.self_attn.in_proj_bias',
        'ar_decoder.layers.7.self_attn.out_proj.weight',
        'ar_decoder.layers.7.self_attn.out_proj.bias',
        'ar_decoder.layers.7.linear1.weight',
        'ar_decoder.layers.7.linear1.bias',
        'ar_decoder.layers.7.linear2.weight',
        'ar_decoder.layers.7.linear2.bias',
        'ar_decoder.layers.7.norm1.weight',
        'ar_decoder.layers.7.norm1.bias',
        'ar_decoder.layers.7.norm2.weight',
        'ar_decoder.layers.7.norm2.bias',
        'ar_decoder.layers.8.self_attn.in_proj_weight',
        'ar_decoder.layers.8.self_attn.in_proj_bias',
        'ar_decoder.layers.8.self_attn.out_proj.weight',
        'ar_decoder.layers.8.self_attn.out_proj.bias',
        'ar_decoder.layers.8.linear1.weight',
        'ar_decoder.layers.8.linear1.bias',
        'ar_decoder.layers.8.linear2.weight',
        'ar_decoder.layers.8.linear2.bias',
        'ar_decoder.layers.8.norm1.weight',
        'ar_decoder.layers.8.norm1.bias',
        'ar_decoder.layers.8.norm2.weight',
        'ar_decoder.layers.8.norm2.bias',
        'ar_decoder.layers.9.self_attn.in_proj_weight',
        'ar_decoder.layers.9.self_attn.in_proj_bias',
        'ar_decoder.layers.9.self_attn.out_proj.weight',
        'ar_decoder.layers.9.self_attn.out_proj.bias',
        'ar_decoder.layers.9.linear1.weight',
        'ar_decoder.layers.9.linear1.bias',
        'ar_decoder.layers.9.linear2.weight',
        'ar_decoder.layers.9.linear2.bias',
        'ar_decoder.layers.9.norm1.weight',
        'ar_decoder.layers.9.norm1.bias',
        'ar_decoder.layers.9.norm2.weight',
        'ar_decoder.layers.9.norm2.bias',
        'ar_decoder.layers.10.self_attn.in_proj_weight',
        'ar_decoder.layers.10.self_attn.in_proj_bias',
        'ar_decoder.layers.10.self_attn.out_proj.weight',
        'ar_decoder.layers.10.self_attn.out_proj.bias',
        'ar_decoder.layers.10.linear1.weight',
        'ar_decoder.layers.10.linear1.bias',
        'ar_decoder.layers.10.linear2.weight',
        'ar_decoder.layers.10.linear2.bias',
        'ar_decoder.layers.10.norm1.weight',
        'ar_decoder.layers.10.norm1.bias',
        'ar_decoder.layers.10.norm2.weight',
        'ar_decoder.layers.10.norm2.bias',
        'ar_decoder.layers.11.self_attn.in_proj_weight',
        'ar_decoder.layers.11.self_attn.in_proj_bias',
        'ar_decoder.layers.11.self_attn.out_proj.weight',
        'ar_decoder.layers.11.self_attn.out_proj.bias',
        'ar_decoder.layers.11.linear1.weight',
        'ar_decoder.layers.11.linear1.bias',
        'ar_decoder.layers.11.linear2.weight',
        'ar_decoder.layers.11.linear2.bias',
        'ar_decoder.layers.11.norm1.weight',
        'ar_decoder.layers.11.norm1.bias',
        'ar_decoder.layers.11.norm2.weight',
        'ar_decoder.layers.11.norm2.bias',
        'ar_decoder.norm.weight',
        'ar_decoder.norm.bias',
        'ar_predict_layer.weight',
        'nar_audio_embeddings.0.word_embeddings.weight',
        'nar_audio_embeddings.1.word_embeddings.weight',
        'nar_audio_embeddings.2.word_embeddings.weight',
        'nar_audio_embeddings.3.word_embeddings.weight',
        'nar_audio_embeddings.4.word_embeddings.weight',
        'nar_audio_embeddings.5.word_embeddings.weight',
        'nar_audio_embeddings.6.word_embeddings.weight',
        'nar_audio_embeddings.7.word_embeddings.weight',
        'nar_text_position.alpha',
        'nar_audio_position.alpha',
        'nar_decoder.layers.0.self_attn.in_proj_weight',
        'nar_decoder.layers.0.self_attn.in_proj_bias',
        'nar_decoder.layers.0.self_attn.out_proj.weight',
        'nar_decoder.layers.0.self_attn.out_proj.bias',
        'nar_decoder.layers.0.linear1.weight',
        'nar_decoder.layers.0.linear1.bias',
        'nar_decoder.layers.0.linear2.weight',
        'nar_decoder.layers.0.linear2.bias',
        'nar_decoder.layers.0.norm1.project_layer.weight',
        'nar_decoder.layers.0.norm1.project_layer.bias',
        'nar_decoder.layers.0.norm1.norm.weight',
        'nar_decoder.layers.0.norm1.norm.bias',
        'nar_decoder.layers.0.norm2.project_layer.weight',
        'nar_decoder.layers.0.norm2.project_layer.bias',
        'nar_decoder.layers.0.norm2.norm.weight',
        'nar_decoder.layers.0.norm2.norm.bias',
        'nar_decoder.layers.1.self_attn.in_proj_weight',
        'nar_decoder.layers.1.self_attn.in_proj_bias',
        'nar_decoder.layers.1.self_attn.out_proj.weight',
        'nar_decoder.layers.1.self_attn.out_proj.bias',
        'nar_decoder.layers.1.linear1.weight',
        'nar_decoder.layers.1.linear1.bias',
        'nar_decoder.layers.1.linear2.weight',
        'nar_decoder.layers.1.linear2.bias',
        'nar_decoder.layers.1.norm1.project_layer.weight',
        'nar_decoder.layers.1.norm1.project_layer.bias',
        'nar_decoder.layers.1.norm1.norm.weight',
        'nar_decoder.layers.1.norm1.norm.bias',
        'nar_decoder.layers.1.norm2.project_layer.weight',
        'nar_decoder.layers.1.norm2.project_layer.bias',
        'nar_decoder.layers.1.norm2.norm.weight',
        'nar_decoder.layers.1.norm2.norm.bias',
        'nar_decoder.layers.2.self_attn.in_proj_weight',
        'nar_decoder.layers.2.self_attn.in_proj_bias',
        'nar_decoder.layers.2.self_attn.out_proj.weight',
        'nar_decoder.layers.2.self_attn.out_proj.bias',
        'nar_decoder.layers.2.linear1.weight',
        'nar_decoder.layers.2.linear1.bias',
        'nar_decoder.layers.2.linear2.weight',
        'nar_decoder.layers.2.linear2.bias',
        'nar_decoder.layers.2.norm1.project_layer.weight',
        'nar_decoder.layers.2.norm1.project_layer.bias',
        'nar_decoder.layers.2.norm1.norm.weight',
        'nar_decoder.layers.2.norm1.norm.bias',
        'nar_decoder.layers.2.norm2.project_layer.weight',
        'nar_decoder.layers.2.norm2.project_layer.bias',
        'nar_decoder.layers.2.norm2.norm.weight',
        'nar_decoder.layers.2.norm2.norm.bias',
        'nar_decoder.layers.3.self_attn.in_proj_weight',
        'nar_decoder.layers.3.self_attn.in_proj_bias',
        'nar_decoder.layers.3.self_attn.out_proj.weight',
        'nar_decoder.layers.3.self_attn.out_proj.bias',
        'nar_decoder.layers.3.linear1.weight',
        'nar_decoder.layers.3.linear1.bias',
        'nar_decoder.layers.3.linear2.weight',
        'nar_decoder.layers.3.linear2.bias',
        'nar_decoder.layers.3.norm1.project_layer.weight',
        'nar_decoder.layers.3.norm1.project_layer.bias',
        'nar_decoder.layers.3.norm1.norm.weight',
        'nar_decoder.layers.3.norm1.norm.bias',
        'nar_decoder.layers.3.norm2.project_layer.weight',
        'nar_decoder.layers.3.norm2.project_layer.bias',
        'nar_decoder.layers.3.norm2.norm.weight',
        'nar_decoder.layers.3.norm2.norm.bias',
        'nar_decoder.layers.4.self_attn.in_proj_weight',
        'nar_decoder.layers.4.self_attn.in_proj_bias',
        'nar_decoder.layers.4.self_attn.out_proj.weight',
        'nar_decoder.layers.4.self_attn.out_proj.bias',
        'nar_decoder.layers.4.linear1.weight',
        'nar_decoder.layers.4.linear1.bias',
        'nar_decoder.layers.4.linear2.weight',
        'nar_decoder.layers.4.linear2.bias',
        'nar_decoder.layers.4.norm1.project_layer.weight',
        'nar_decoder.layers.4.norm1.project_layer.bias',
        'nar_decoder.layers.4.norm1.norm.weight',
        'nar_decoder.layers.4.norm1.norm.bias',
        'nar_decoder.layers.4.norm2.project_layer.weight',
        'nar_decoder.layers.4.norm2.project_layer.bias',
        'nar_decoder.layers.4.norm2.norm.weight',
        'nar_decoder.layers.4.norm2.norm.bias',
        'nar_decoder.layers.5.self_attn.in_proj_weight',
        'nar_decoder.layers.5.self_attn.in_proj_bias',
        'nar_decoder.layers.5.self_attn.out_proj.weight',
        'nar_decoder.layers.5.self_attn.out_proj.bias',
        'nar_decoder.layers.5.linear1.weight',
        'nar_decoder.layers.5.linear1.bias',
        'nar_decoder.layers.5.linear2.weight',
        'nar_decoder.layers.5.linear2.bias',
        'nar_decoder.layers.5.norm1.project_layer.weight',
        'nar_decoder.layers.5.norm1.project_layer.bias', 'nar_decoder.layers.5.norm1.norm.weight', 'nar_decoder.layers.5.norm1.norm.bias', 'nar_decoder.layers.5.norm2.project_layer.weight', 'nar_decoder.layers.5.norm2.project_layer.bias', 'nar_decoder.layers.5.norm2.norm.weight', 'nar_decoder.layers.5.norm2.norm.bias', 'nar_decoder.layers.6.self_attn.in_proj_weight', 'nar_decoder.layers.6.self_attn.in_proj_bias', 'nar_decoder.layers.6.self_attn.out_proj.weight', 'nar_decoder.layers.6.self_attn.out_proj.bias', 'nar_decoder.layers.6.linear1.weight', 'nar_decoder.layers.6.linear1.bias', 'nar_decoder.layers.6.linear2.weight', 'nar_decoder.layers.6.linear2.bias', 'nar_decoder.layers.6.norm1.project_layer.weight', 'nar_decoder.layers.6.norm1.project_layer.bias', 'nar_decoder.layers.6.norm1.norm.weight', 'nar_decoder.layers.6.norm1.norm.bias', 'nar_decoder.layers.6.norm2.project_layer.weight', 'nar_decoder.layers.6.norm2.project_layer.bias', 'nar_decoder.layers.6.norm2.norm.weight', 'nar_decoder.layers.6.norm2.norm.bias', 'nar_decoder.layers.7.self_attn.in_proj_weight', 'nar_decoder.layers.7.self_attn.in_proj_bias', 'nar_decoder.layers.7.self_attn.out_proj.weight', 'nar_decoder.layers.7.self_attn.out_proj.bias', 'nar_decoder.layers.7.linear1.weight', 'nar_decoder.layers.7.linear1.bias', 'nar_decoder.layers.7.linear2.weight', 'nar_decoder.layers.7.linear2.bias', 'nar_decoder.layers.7.norm1.project_layer.weight', 'nar_decoder.layers.7.norm1.project_layer.bias', 'nar_decoder.layers.7.norm1.norm.weight', 'nar_decoder.layers.7.norm1.norm.bias', 'nar_decoder.layers.7.norm2.project_layer.weight', 'nar_decoder.layers.7.norm2.project_layer.bias', 'nar_decoder.layers.7.norm2.norm.weight', 'nar_decoder.layers.7.norm2.norm.bias', 'nar_decoder.layers.8.self_attn.in_proj_weight', 'nar_decoder.layers.8.self_attn.in_proj_bias', 'nar_decoder.layers.8.self_attn.out_proj.weight', 'nar_decoder.layers.8.self_attn.out_proj.bias', 'nar_decoder.layers.8.linear1.weight', 'nar_decoder.layers.8.linear1.bias', 'nar_decoder.layers.8.linear2.weight', 'nar_decoder.layers.8.linear2.bias', 'nar_decoder.layers.8.norm1.project_layer.weight', 'nar_decoder.layers.8.norm1.project_layer.bias', 'nar_decoder.layers.8.norm1.norm.weight', 'nar_decoder.layers.8.norm1.norm.bias', 'nar_decoder.layers.8.norm2.project_layer.weight', 'nar_decoder.layers.8.norm2.project_layer.bias', 'nar_decoder.layers.8.norm2.norm.weight', 'nar_decoder.layers.8.norm2.norm.bias', 'nar_decoder.layers.9.self_attn.in_proj_weight', 'nar_decoder.layers.9.self_attn.in_proj_bias', 'nar_decoder.layers.9.self_attn.out_proj.weight', 'nar_decoder.layers.9.self_attn.out_proj.bias', 'nar_decoder.layers.9.linear1.weight', 'nar_decoder.layers.9.linear1.bias', 'nar_decoder.layers.9.linear2.weight', 'nar_decoder.layers.9.linear2.bias', 'nar_decoder.layers.9.norm1.project_layer.weight', 'nar_decoder.layers.9.norm1.project_layer.bias', 'nar_decoder.layers.9.norm1.norm.weight', 'nar_decoder.layers.9.norm1.norm.bias', 'nar_decoder.layers.9.norm2.project_layer.weight', 'nar_decoder.layers.9.norm2.project_layer.bias', 'nar_decoder.layers.9.norm2.norm.weight', 'nar_decoder.layers.9.norm2.norm.bias', 'nar_decoder.layers.10.self_attn.in_proj_weight', 'nar_decoder.layers.10.self_attn.in_proj_bias', 'nar_decoder.layers.10.self_attn.out_proj.weight', 'nar_decoder.layers.10.self_attn.out_proj.bias', 'nar_decoder.layers.10.linear1.weight', 'nar_decoder.layers.10.linear1.bias', 'nar_decoder.layers.10.linear2.weight', 'nar_decoder.layers.10.linear2.bias', 'nar_decoder.layers.10.norm1.project_layer.weight', 'nar_decoder.layers.10.norm1.project_layer.bias', 'nar_decoder.layers.10.norm1.norm.weight', 'nar_decoder.layers.10.norm1.norm.bias', 'nar_decoder.layers.10.norm2.project_layer.weight', 'nar_decoder.layers.10.norm2.project_layer.bias', 'nar_decoder.layers.10.norm2.norm.weight', 'nar_decoder.layers.10.norm2.norm.bias', 'nar_decoder.layers.11.self_attn.in_proj_weight', 'nar_decoder.layers.11.self_attn.in_proj_bias', 'nar_decoder.layers.11.self_attn.out_proj.weight', 'nar_decoder.layers.11.self_attn.out_proj.bias', 'nar_decoder.layers.11.linear1.weight', 'nar_decoder.layers.11.linear1.bias', 'nar_decoder.layers.11.linear2.weight', 'nar_decoder.layers.11.linear2.bias', 'nar_decoder.layers.11.norm1.project_layer.weight', 'nar_decoder.layers.11.norm1.project_layer.bias', 'nar_decoder.layers.11.norm1.norm.weight', 'nar_decoder.layers.11.norm1.norm.bias', 'nar_decoder.layers.11.norm2.project_layer.weight', 'nar_decoder.layers.11.norm2.project_layer.bias', 'nar_decoder.layers.11.norm2.norm.weight', 'nar_decoder.layers.11.norm2.norm.bias', 'nar_decoder.norm.project_layer.weight', 'nar_decoder.norm.project_layer.bias', 'nar_decoder.norm.norm.weight', 'nar_decoder.norm.norm.bias', 'nar_predict_layers.0.weight', 'nar_predict_layers.1.weight', 'nar_predict_layers.2.weight', 'nar_predict_layers.3.weight', 'nar_predict_layers.4.weight', 'nar_predict_layers.5.weight', 'nar_predict_layers.6.weight', 'nar_stage_embeddings.0.word_embeddings.weight', 'nar_stage_embeddings.1.word_embeddings.weight', 'nar_stage_embeddings.2.word_embeddings.weight', 'nar_stage_embeddings.3.word_embeddings.weight', 'nar_stage_embeddings.4.word_embeddings.weight', 'nar_stage_embeddings.5.word_embeddings.weight', 'nar_stage_embeddings.6.word_embeddings.weight', 'ar_language_embedding.word_embeddings.weight', 'nar_language_embedding.word_embeddings.weight'])

class VALLF_Model {
    ar_embeddings *
};

class VALLF {
public:
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
    int nhead;
    int prefix_mode;
    int num_quantizers;

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
            int num_quantizers = 8) {

        nar_d_model = int(d_model * nar_scale_factor);
        ar_text_embedding = new TokenEmbedding(d_model, num_token);
        nar_text_embedding = new TokenEmbedding(nar_d_model, num_token);
        ar_audio_prepend_bos = prepend_bos;

        if (add_prenet) {
            /* code */
        } else {
            ar_text_prenet = nullptr;
            nar_audio_prenet = nullptr;
        }

        ar_text_position = new SinePositionalEmbedding(d_model,
                                                       0.1,
                                                       false,
                                                       true);
        ar_audio_position = new SinePositionalEmbedding(nar_d_model,
                                                        0.1,
                                                        false,
                                                        true);
        ar_decoder = new TransformerDecoder();
        ar_predict_layer = new TransformerDecoderLayer();
//        rng = random.Random(0)
        nhead = nhead;
        prefix_mode = prefix_mode;
        num_quantizers = num_quantizers;
    }

    load_model_from_file()
};

class VALLE {
};