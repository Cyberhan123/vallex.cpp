#include "common.h"

#include "ggml/ggml.h"
#include "ggml/ggml-backend.h"

#include <vector>

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

struct ggml_tensor *ggml_arange(struct ggml_graph_ctx *ctx, enum ggml_type type, int start, int end,int step) {
    const auto x = ggml_new_tensor_1d(ctx->ctx, type, 1);
    if (!ggml_allocr_is_measure(ctx->allocr)) {
        std::vector<float> vec;
        for (int i = start; i < end; i+=step) {
            vec.push_back(static_cast<float >(i));
        };
        ggml_backend_tensor_set(x, vec.data(), 0, vec.size() * sizeof(int));
    }
    return x;
}

struct ggml_tensor *ggml_arange_2d(struct ggml_graph_ctx *ctx, enum ggml_type type, int start, int end,int step) {
    const auto x = ggml_new_tensor_2d(ctx->ctx, type, 1, 1);
    if (!ggml_allocr_is_measure(ctx->allocr)) {
        std::vector<std::vector<float>> vec;
        for (int i = start; i < end; i+=step) {
            std::vector<float> each;
            each.push_back(static_cast<float >(i));
            vec.push_back(each);
        };

        ggml_backend_tensor_set(x, vec.data(), 0, vec.size() * sizeof(int));
    }
    return x;
};

//LifeCycle::LifeCycle(ggml_init_params params) {
//    // init ggml_time total
//    ggml_time_init();
//    this->t_start_us = ggml_time_us();
//    // create ggml ctx
//    this->ctx = ggml_init(params);
//
//}
//
//bool LifeCycle::to_gpu() {
//    if (this->backend == nullptr) {
//        // create ggml gpu backend
//#ifdef GGML_USE_CUBLAS
//        this->backend = ggml_backend_cuda_init();
//#endif
//#ifdef GGML_USE_METAL
//        this->backend = ggml_backend_metal_init();
//#endif
//    }
//
//    if (this->backend != nullptr) {
//        return true;
//    }
//    return false;
//}
//
//bool LifeCycle::to_cpu() {
//    if (this->backend == nullptr) {
//        this->backend = ggml_backend_cpu_init();
//        return true;
//    }
//    return false;
//}
//
//template<typename... Ts>
//bool LifeCycle::register_as(NN::Module<Ts> module, std::string name) {
//    return 0;
//}
