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


struct ggml_tensor *
ggml_vallex_arange_1d(struct vallex_compute_context *ctx, int start, int end, int step) {
    const auto tensor = ggml_new_tensor_1d(ctx->context, GGML_TYPE_F32, end);
    ggml_allocr_alloc(ctx->allocr, tensor);
    if (!ggml_allocr_is_measure(ctx->allocr)) {
        std::vector<float> vec;

        for (int i = start; i < end; i += step) {
            vec.push_back(static_cast<float >(i));
        };
        ggml_backend_tensor_set(tensor, vec.data(), 0, ggml_nbytes(tensor));
    }
    return tensor;
}

struct ggml_tensor *
ggml_vallex_arange_2d(struct vallex_compute_context *ctx, int start, int end, int step) {
    const auto tensor = ggml_new_tensor_2d(ctx->context, GGML_TYPE_F32, end + 1, 1);
    ggml_allocr_alloc(ctx->allocr, tensor);
    if (!ggml_allocr_is_measure(ctx->allocr)) {
        std::vector<std::vector<float>> vec;

        for (int i = start; i < end; i += step) {
            std::vector<float> each;
            each.push_back(static_cast<float >(i));
            vec.push_back(each);
        };
        ggml_backend_tensor_set(tensor, vec.data(), 0, ggml_nbytes(tensor));
    }
    return tensor;
}

static void ggml_vallex_sin(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata) {
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
        dst_data[i] = sinf(src_data[i]);
    }
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

    const int ne = (int) ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
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

//struct ggml_tensor *ggml