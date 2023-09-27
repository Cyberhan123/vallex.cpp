
#include "vallex.h"

#include "embedding.h"
#include "ggml.h"

class TransformerDecoder
{
};

class TransformerDecoderLayer
{
};

class VALLF
{
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
    
    VALLF(
        int d_model,
        int nhead,
        int num_layers,
        bool norm_first = true,
        bool add_prenet = false,
        TransformerDecoder *decoder_cls = NULL,
        TransformerDecoderLayer *decoder_layer_cls = NULL,
        int prefix_mode = 0,
        bool share_embedding = true,
        float nar_scale_factor = 1.0,
        bool prepend_bos = true,
        int num_token = 1024,
        int num_quantizers = 8)
    {

        nar_d_model = int(d_model * nar_scale_factor);
        ar_text_embedding = new TokenEmbedding(d_model, num_token);
        nar_text_embedding = new TokenEmbedding(nar_d_model, num_token);
        ar_audio_prepend_bos = prepend_bos;

        if (add_prenet)
        {
            /* code */
        }
        else
        {
            ar_text_prenet = NULL;
            nar_audio_prenet = NULL;
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

    }
};

class VALLE
{
};