#pragma once
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <cmath>

#include <ggml/ggml.h>

class SinePositionalEmbedding {
public:
	SinePositionalEmbedding(
		int d_model,
		float dropout = 0.0,
		bool scale = false,
		bool normalize = false
	);

private:
	int dim_model;
	float x_scale;
	ggml_tensor* alpha;
	void* dropout;
	bool reverse;
	ggml_tensor* pe;
};

class TokenEmbedding {
public:
	TokenEmbedding(int d_model, int num_token, float dropout = 0.0);

	void set_data(ggml_tensor* data);

	ggml_tensor* forward(ggml_tensor* x);

protected:
	int d_model;
	int num_token;
	float dropout;
	ggml_tensor* data;
};

#endif