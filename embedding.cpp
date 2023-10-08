#include "embedding.h"
#include "embedding.h"

SinePositionalEmbedding::SinePositionalEmbedding(
	int d_model,
	float dropout = 0.0,
	bool scale = false,
	bool normalize = false
)
{
	dim_model = dim_model
		if (scale)
		{
			x_scale = std::sqrt(dim_model)
		}
		else {
			x_scale = 1.0
		}
}

TokenEmbedding::TokenEmbedding(int d_model, int num_token, float dropout) {
	d_model = d_model;
	num_token = num_token;
	dropout = dropout;
}

void TokenEmbedding::set_data(ggml_tensor* view) {
	data = view;
}

ggml_tensor* TokenEmbedding::forward(ggml_tensor* x)
{
	return nullptr;
}
