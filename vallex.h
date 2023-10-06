#pragma once
#ifndef VALLEX_H
#define VALLEX_H

#include <string>

#include "embedding.h"
#include "transformer.h"
#include "ggml.h"

class VALLF {
public:
	

	VALLF(
		int d_model,
		int nhead,
		int num_layers,
		bool norm_first = true,
		bool add_prenet = false,
		TransformerDecoder* decoder_cls = nullptr,
		TransformerDecoderLayer* decoder_layer_cls = nullptr,
		int prefix_mode = 0,
		bool share_embedding = true,
		float nar_scale_factor = 1.0,
		bool prepend_bos = true,
		int num_token = 1024,
		int num_quantizers = 8);

	bool load_model_from_file();


private:
	int nar_d_model;
	TokenEmbedding* ar_text_embedding;
	TokenEmbedding* nar_text_embedding;
	bool ar_audio_prepend_bos;

	void* ar_text_prenet;
	void* nar_audio_prenet;

	SinePositionalEmbedding* ar_text_position;
	SinePositionalEmbedding* ar_audio_position;

	TransformerDecoder* ar_decoder;
	TransformerDecoderLayer* ar_predict_layer;
	int num_heads;
	int prefix_mode;
};


class VALLE : public VALLF {
public:
	VALLE(
		int d_model,
		int nhead,
		int num_layers,
		bool norm_first = true,
		bool add_prenet = false,
		TransformerDecoder* decoder_cls = nullptr,
		TransformerDecoderLayer* decoder_layer_cls = nullptr,
		int prefix_mode = 0,
		bool share_embedding = true,
		float nar_scale_factor = 1.0,
		bool prepend_bos = true,
		int num_token = 1024,
		int num_quantizers = 8);

	ggml_tensor* inference(
		ggml_tensor* x,
		ggml_tensor* x_lens,
		ggml_tensor* y,
		ggml_tensor* enroll_x_lens,
		int	top_k = -100,
		float temperature = 1.0,
		std::string prompt_language,
		std::string	text_language,
		int best_of = 1,
		float	length_penalty = 1.0,
		bool return_worst = false,
		)
};

#endif