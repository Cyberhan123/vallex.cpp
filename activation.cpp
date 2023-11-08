#include "activation.h"

MultiheadAttention::MultiheadAttention(
        int64_t embed_dim,
        int64_t num_heads,
        float dropout,
        bool bias,
        bool add_bias_kv,
        bool add_zero_attn,
        int64_t kdim,
        int64_t vdim,
        bool batch_first,
        ggml_tensor *linear1_cls,
        ggml_tensor *linear2_cls,
        ggml_type dtype
) {
    this->embed_dim = embed_dim;
    this->num_heads = num_heads;
}

size_t MultiheadAttention::compute_params_mem_size(ggml_type wtype) {
    float mem_size = static_cast<float > (this->num_heads * this->embed_dim) * ggml_type_sizef(wtype);
    return static_cast<size_t>(mem_size);

}

void MultiheadAttention::init_params(struct ggml_context *ctx, ggml_type wtype) {
    this->in_proj_weight = ggml_new_tensor_2d(ctx, wtype, this->embed_dim, this->num_heads);
    this->in_proj_bias = ggml_new_tensor_1d(ctx, wtype, this->embed_dim);
    this->out_proj_weight = ggml_new_tensor_2d(ctx, wtype, this->embed_dim, this->num_heads);
    this->out_proj_bias = ggml_new_tensor_1d(ctx, wtype, this->embed_dim);
}

void MultiheadAttention::mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) {
    tensors[prefix + ".in_proj_weight"] = this->in_proj_weight;
    tensors[prefix + ".in_proj_bias"] = this->in_proj_weight;
    tensors[prefix + ".out_proj_weight"] = this->in_proj_weight;
    tensors[prefix + ".out_proj_bias"] = this->in_proj_weight;
}

struct ggml_vallex_split_tensor_2d_params {
    int64_t ne0[2]; //start, end
    int64_t ne1[2]; //start, end
};

static void
ggml_vallex_split_tensor_2d(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith,
                            int nth, void *userdata) {
    GGML_ASSERT(userdata == nullptr);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));


    float *sin_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    GGML_TENSOR_LOCALS(int64_t, sin_ne, src, ne)
    GGML_TENSOR_LOCALS(size_t, sin_nb, src, nb)

    GGML_TENSOR_LOCALS(int64_t, dst_ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, dst_nb, dst, nb)

    const auto ne = ggml_nelements(dst);
    const auto dr = (ne + nth - 1) / nth;
    const auto ie0 = dr * ith;
    const auto ie1 = std::min(ie0 + dr, ne);
    const ggml_vallex_split_tensor_2d_params *params = (ggml_vallex_split_tensor_2d_params *) userdata;
    for (auto i = ie0; i < ie1; ++i) {
        /*const int row = i / dst_ne0; //row index*/
        const auto column = i % dst_ne0; //column index
        //TODO

//        if (column == 0 || column % 2 == 0) {
//            dst_data[i] = sin_data[column];
//        } else {
//            dst_data[i] = cos_data[column];
//        }
    }
    dst->ne[0] = params->ne0[1] - params->ne0[0];
    dst->ne[1] = params->ne1[1] - params->ne1[0];
}

struct ggml_tensor *
MultiheadAttention::forward(
        vallex_compute_context *ctx,
        ggml_tensor *x,
        ggml_tensor *key_padding_mask,
        bool need_weights,
        ggml_tensor *attn_mask,
        bool average_attn_weights,
        ggml_tensor *past_kv,
        bool use_cache,
        ggml_tensor *k_cache,
        ggml_tensor *v_cache
) {
    const auto B = x->ne[0];
    const auto T = x->ne[1];
    const auto C = x->ne[2];
    auto cur = ggml_vallex_linear(ctx->context, x, this->in_proj_weight, this->in_proj_bias);
    const auto cur_total = ggml_nrows(cur) / 3;

    //    q, k, v = torch._C._nn.linear(x, ipw, ipb).chunk(3, dim=-1)
    //    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    //    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    //    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    const auto ne0 = B;
    const auto ne1 = T;
    const auto ne2 = this->num_heads;
    const auto ne3 = C / this->num_heads;
    const auto nb1 = cur->nb[0] * (ne0 / ggml_blck_size(cur->type));
    const auto nb2 = cur->nb[1] * ne1;
    const auto nb3 = cur->nb[2] * ne2;

    auto Q = ggml_view_4d(ctx->context, cur, ne0, ne1, ne2, ne3, nb1, nb2, nb3, 0 * sizeof(float) * cur_total);
    Q = ggml_permute(ctx->context, Q, 0, 2, 1, 3);


    auto K = ggml_view_4d(ctx->context, cur, ne0, ne1, ne2, ne3, nb1, nb2, nb3, 1 * sizeof(float) * cur_total);
    K = ggml_permute(ctx->context, K, 0, 2, 1, 3);

    auto V = ggml_view_4d(ctx->context, cur, ne0, ne1, ne2, ne3, nb1, nb2, nb3, 2 * sizeof(float) * cur_total);
    V = ggml_permute(ctx->context, V, 0, 2, 1, 3);

    if (past_kv != nullptr) {
        //    past_key = past_kv[0]
        //    past_value = past_kv[1]
        //    k = torch.cat((past_key, k), dim=-2)
        //    v = torch.cat((past_value, v), dim=-2)
        const auto past_key = ggml_view_1d(ctx->context, past_kv, 1, 0);
        const auto past_value = ggml_view_1d(ctx->context, past_kv, 1, sizeof(float));
        //TODO: can we use ggml_cpy ??
        K = ggml_vallex_cat(ctx->context, past_key, K, -2);
        V = ggml_vallex_cat(ctx->context, past_value, V, -2);
        //TODO: need call ggml_build_forward_expand ?

        // ggml_build_forward_expand(gf,K);
        // ggml_build_forward_expand(gf,V);
    }
    const auto FULL_T = K->ne[K->n_dims - 2];

    if (use_cache) {
        // should use view or not?
        k_cache = ggml_view_tensor(ctx->context, K);
        v_cache = ggml_view_tensor(ctx->context, V);
    }
    //    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    //    att = att.masked_fill(attn_mask[FULL_T - T:FULL_T, :FULL_T], float('-inf'))
    //    att = F.softmax(att, dim=-1)
    // TODO confirm  K shape
    auto attn = ggml_vallex_mul_num(
            ctx->context,
            ggml_mul(ctx->context, Q, ggml_permute(ctx->context, K, 0, 2, 1, 3)),
            1.0f / sqrtf(K->ne[K->n_dims - 2])
    );

    const auto split_params = new ggml_vallex_split_tensor_2d_params{
            {FULL_T - T, FULL_T},
            {0,          FULL_T}
    };

    attn_mask = ggml_map_custom1(ctx->context,
                                 attn_mask,
                                 ggml_vallex_split_tensor_2d,
                                 GGML_N_TASKS_MAX,
                                 split_params);

    attn = ggml_vallex_masked_fill(ctx->context, attn, attn_mask, -INFINITY);
    attn = ggml_soft_max(ctx->context, attn);
    attn = ggml_reshape_1d(ctx->context, attn, ggml_nelements(attn));

    // y = att @ v
    // y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    // y = torch._C._nn.linear(y, opw, opb)

    auto y = ggml_mul(ctx->context, attn, V);
    y = ggml_view_3d(
            ctx->context,
            ggml_cont(
                    ctx->context,
                    ggml_permute(ctx->context, y, 0, 2, 1, 3)),
            B, T, C,
            y->nb[1], y->nb[2],
            0
    );
    y = ggml_vallex_linear(ctx->context, y, this->out_proj_weight, this->out_proj_bias);

    return y;
}
