#include <algorithm>
#include <cmath>
#include <random>
#include "common.h"

VallexGlobalContext::VallexGlobalContext(size_t n_tensors, size_t buffer_size) {
    {
        struct ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
        };
        this->context = ggml_init(params);
        if (!this->context) {
            LOG_ERROR("%s: ggml_init() failed\n", __func__);
        }
    }

    // initialize the backend
#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > 0) {
        LOG_INFO("%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init();
        if (!model.backend) {
            LOG_ERROR("%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        LOG_INFO(stderr, "%s: using Metal backend\n", __func__);
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
        this->backend = ggml_backend_metal_init();
        if (!model.backend) {
            LOG_ERROR( "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if (!this->backend) {
        // fallback to CPU backend
        LOG_INFO("%s: using CPU backend\n", __func__);
        this->backend = ggml_backend_cpu_init();
    }

    if (!this->backend) {
        LOG_ERROR("%s: ggml_backend_cpu_init() failed\n", __func__);
    }

    // allocate weights buffer
    this->buffer = ggml_backend_alloc_buffer(this->backend, buffer_size);
}

struct vallex_arange_params {
    int64_t start;
    int64_t end;
    int step;
};

static void
ggml_vallex_arange_impl(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata) {
    GGML_ASSERT(userdata != nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

//    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const int ne = (int) ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);
    const vallex_arange_params *params = (vallex_arange_params *) userdata;

    for (auto i = ie0; i < ie1; ++i) {
        if (ie0 >= params->start && ie0 <= params->end) {
            const auto r = ie0 * params->step;
            dst_data[i] = static_cast<float>(r);
        }
    }
}


struct ggml_tensor *
ggml_vallex_arange(struct ggml_context *ctx, struct ggml_tensor *a, int64_t start, int64_t end, int step) {
    const auto userdata = new vallex_arange_params{
            start,
            end,
            step
    };
    return ggml_map_custom1(ctx, a, ggml_vallex_arange_impl, GGML_N_TASKS_MAX, userdata);
}


static void
ggml_vallex_exp_impl(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata) {
    GGML_ASSERT(userdata == nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const int ne = (int) ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = expf(src_data[i]);
    }
}

struct ggml_tensor *ggml_vallex_exp(struct ggml_context *ctx, struct ggml_tensor *a) {
    return ggml_map_custom1(ctx, a, ggml_vallex_exp_impl, GGML_N_TASKS_MAX, nullptr);
}

static void
ggml_vallex_sin_impl(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata) {
    GGML_ASSERT(userdata == nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const auto ne = (int) ggml_nelements(dst);
    const auto dr = (ne + nth - 1) / nth;
    const auto ie0 = dr * ith;
    const auto ie1 = std::min(ie0 + dr, ne);

    for (auto i = ie0; i < ie1; ++i) {
        dst_data[i] = sinf(src_data[i]);
    }
}

struct ggml_tensor *ggml_vallex_sin(struct ggml_context *ctx, struct ggml_tensor *a) {
    return ggml_map_custom1(ctx, a, ggml_vallex_sin_impl, GGML_N_TASKS_MAX, nullptr);
}

static void
ggml_vallex_cos_impl(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata) {
    GGML_ASSERT(userdata == nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const int ne = (int) ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = cosf(src_data[i]);
    }
}

struct ggml_tensor *ggml_vallex_cos(struct ggml_context *ctx, struct ggml_tensor *a) {
    return ggml_map_custom1(ctx, a, ggml_vallex_cos_impl, GGML_N_TASKS_MAX, nullptr);
}

struct ggml_vallex_mul_num_params {
    float num;
};

static void
ggml_vallex_mul_num_impl(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata) {
    GGML_ASSERT(userdata != nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const int ne = (int) ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    const ggml_vallex_mul_num_params *params = (ggml_vallex_mul_num_params *) userdata;

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = src_data[i] * params->num;
    }
}

struct ggml_tensor *ggml_vallex_mul_num(struct ggml_context *ctx, struct ggml_tensor *a, float b) {
    const auto userdata = new ggml_vallex_mul_num_params{
            b
    };
    return ggml_map_custom1(ctx, a, ggml_vallex_mul_num_impl, GGML_N_TASKS_MAX, userdata);
}

struct ggml_tensor *ggml_vallex_linear(
        struct ggml_context *ctx,
        struct ggml_tensor *input,
        struct ggml_tensor *weight,
        struct ggml_tensor *bias
) {
    auto output = ggml_mul_mat(ctx, input, weight);
    if (bias != nullptr) {
        output = ggml_add(ctx, output, bias);
    }
    return output;
};


struct ggml_vallex_masked_fill_params {
    float value;
};


static void
ggml_vallex_masked_fill_impl(struct ggml_tensor *dst, const struct ggml_tensor *a, const struct ggml_tensor *b, int ith,
                             int nth, void *userdata) {
    GGML_ASSERT(userdata != nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, a));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));

    const float *a_data = ggml_get_data_f32(a);
    const float *b_data = ggml_get_data_f32(b);
    float *dst_data = ggml_get_data_f32(dst);

    const int a_ne = (int) ggml_nelements(dst);
    const int a_dr = (a_ne + nth - 1) / nth;
    const int a_ie0 = a_dr * ith;
    const int a_ie1 = std::min(a_ie0 + a_dr, a_ne);

    const ggml_vallex_masked_fill_params *params = (ggml_vallex_masked_fill_params *) userdata;

    //TODO: check b shape if a,b are not same shape ,this code don't work.

    for (int i = a_ie0; i < a_ie1; ++i) {
        if (b_data[i] == 0.0f) {
            dst_data[i] = params->value;
            continue;
        }
        dst_data[i] = a_data[i];
    }
}

struct ggml_tensor *
ggml_vallex_masked_fill(struct ggml_context *ctx, struct ggml_tensor *input, struct ggml_tensor *mask, float value) {
    const auto userdata = new ggml_vallex_masked_fill_params{
            value
    };
    return ggml_map_custom2(ctx, input, mask, ggml_vallex_masked_fill_impl, GGML_N_TASKS_MAX, userdata);
}

static void
ggml_vallex_cat_impl(struct ggml_tensor *dst, const struct ggml_tensor *src0, const struct ggml_tensor *src1, int ith,
                     int nth, void *userdata) {

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));


    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);
    GGML_TENSOR_LOCALS(size_t, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    GGML_TENSOR_LOCALS(size_t, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb);

    auto dim = *(int *) userdata;

    const float *src0_data = ggml_get_data_f32(src0);
    const float *src1_data = ggml_get_data_f32(src1);
    float *dst_data = ggml_get_data_f32(dst);

    // rest shape
    dst->ne[dim] = src0->ne[dim] + src1->ne[dim];
    // rest nb
    dst->nb[0] = ggml_type_size(dst->type);
    dst->nb[1] = dst->nb[0] * (dst->ne[0] / ggml_blck_size(dst->type));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        dst->nb[i] = dst->nb[i - 1] * dst->ne[i - 1];
    }


    switch (src0->n_dims) {
        case 1:
            GGML_ASSERT(dim == 0);

            //TODO
            break;
        case 2:
            GGML_ASSERT(dim <= 1);
            //TODO
            break;
        case 3:
            GGML_ASSERT(dim <= 2);
            //TODO
            break;
        case 4:
            GGML_ASSERT(dim <= 3);
            //TODO
            break;
    }
//    for (int i3 = 0; i3 < ne3; i3++) {
//        for (int i2 = ith; i2 < ne2; i2++) {
//            if (i2 < ne02) { // src0
//                for (int i1 = 0; i1 < ne1; i1++) {
//                    for (int i0 = 0; i0 < ne0; i0++) {
//                        const float *x = (float *) ((char *) src0->data + i0 * nb00 + i1 * nb01 + i2 * nb02 +
//                                                    i3 * nb03);
//
//                        float *y = (float *) ((char *) dst->data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
//                        *y = *x;
//                    }
//                }
//            } // src1
//            else {
//                for (int i1 = 0; i1 < ne1; i1++) {
//                    for (int i0 = 0; i0 < ne0; i0++) {
//                        const float *x = (float *) ((char *) src1->data + i0 * nb10 + i1 * nb11 + (i2 - ne02) * nb12 +
//                                                    i3 * nb13);
//
//                        float *y = (float *) ((char *) dst->data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
//                        *y = *x;
//                    }
//                }
//            }
//        }
//    }
}

struct ggml_tensor *
ggml_vallex_cat(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, int dim) {

    if (dim < 0) {
        dim = a->n_dims + dim;
    }
    GGML_ASSERT(a->n_dims >= dim);
    GGML_ASSERT(a->n_dims == b->n_dims);
    // check shape
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i == dim) continue;
        GGML_ASSERT(a->ne[i] == b->ne[i]);
    }
    GGML_ASSERT(dim <= GGML_MAX_DIMS);
    GGML_ASSERT(-GGML_MAX_DIMS <= dim);
    return ggml_map_custom2(ctx, a, b, ggml_vallex_masked_fill_impl, GGML_N_TASKS_MAX, &dim);
};

struct ggml_tensor *ggml_vallex_layer_norm(
        struct ggml_context *ctx,
        struct ggml_tensor *input,
        struct ggml_tensor *normalized_shape,
        struct ggml_tensor *weight,
        struct ggml_tensor *bias,
        float eps
) {
    auto x = ggml_norm(ctx, input, eps);

    if (normalized_shape) {
        x = ggml_reshape(ctx, x, normalized_shape);
    }

    if (weight) {
        x = ggml_mul(ctx, x, weight);
    }

    if (bias) {
        x = ggml_add(ctx, x, bias);
    }

    return x;
}

static void
ggml_vallex_dropout_impl(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata) {
    GGML_ASSERT(userdata != nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const int ne = (int) ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    const float p = *((float *) userdata);

    const ggml_vallex_mul_num_params *params = (ggml_vallex_mul_num_params *) userdata;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = ie0; i < ie1; ++i) {
        auto randNum = distribution(generator);
        if (randNum < p) {
            dst_data[i] = 0.0;
        } else {
            dst_data[i] = src_data[i];
        }
    }
}

struct ggml_tensor *ggml_vallex_dropout(struct ggml_context *ctx, struct ggml_tensor *src, float p) {
    return ggml_map_custom1(ctx, src, ggml_vallex_dropout_impl, GGML_N_TASKS_MAX, &p);
};
