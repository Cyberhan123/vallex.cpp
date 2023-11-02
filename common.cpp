#include <algorithm>
#include <cmath>
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
            dst_data[i] = static_cast<float>(r) ;
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

//struct ggml_tensor *ggml