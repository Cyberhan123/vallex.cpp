#pragma once

#ifndef COMMON_H
#define COMMON_H

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#include <string>
#include <map>
#include <memory>

enum LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

static auto log_level = LogLevel::INFO;

#define FILENAME_ "vallex.cpp"


#define LOGGER(level, format, ...)                                                                    \
    do {                                                                                              \
        if (level < log_level) {                                                                      \
            break;                                                                                    \
        }                                                                                             \
        if (level == LogLevel::DEBUG) {                                                             \
            printf("[DEBUG] %s:%-4d - " format "\n", FILENAME_, __LINE__, ##__VA_ARGS__);          \
            fflush(stdout);                                                                           \
        } else if (level == LogLevel::INFO) {                                                       \
            printf("[INFO]  %s:%-4d - " format "\n", FILENAME_, __LINE__, ##__VA_ARGS__);          \
            fflush(stdout);                                                                           \
        } else if (level == LogLevel::WARN) {                                                       \
            fprintf(stderr, "[WARN]  %s:%-4d - " format "\n", FILENAME_, __LINE__, ##__VA_ARGS__); \
            fflush(stdout);                                                                           \
        } else if (level == LogLevel::ERROR) {                                                      \
            fprintf(stderr, "[ERROR] %s:%-4d - " format "\n", FILENAME_, __LINE__, ##__VA_ARGS__); \
            fflush(stdout);                                                                           \
        }                                                                                             \
    } while (0)

#define LOG_DEBUG(format, ...) LOGGER(LogLevel::DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) LOGGER(LogLevel::INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) LOGGER(LogLevel::WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) LOGGER(LogLevel::ERROR, format, ##__VA_ARGS__)


/**
 * The model must have two step alloc buffer for model weight
 * and alloc buffer for compute,the compute buffer can be rest but the model weight can't
 * So in ggml they have different buffer and allocr, but they use same backend.
 * */

// hold model buffer
class VallexGlobalContext {
public:
    VallexGlobalContext(size_t n_tensors, size_t buffer_size);

    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_context *context;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct vallex_compute_context {
    struct ggml_context *context;
    struct ggml_allocr *allocr;
};

namespace NN {
    template<typename TReturn,typename... TArgs>
    class Module {
        // compute module buff size
        virtual size_t compute_params_mem_size(ggml_type wtype) = 0;

        // init tensor struct
        virtual void init_params(struct ggml_context *ctx, ggml_type wtype) = 0;

        // mapping tensor struct to buff
        virtual void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) = 0;

        virtual TReturn forward(vallex_compute_context *ctx, TArgs... Args) = 0;
    };
}

std::string get_system_info();

struct ggml_tensor *
ggml_vallex_arange(struct ggml_context *ctx, struct ggml_tensor *a, int64_t start, int64_t end, int step);

struct ggml_tensor *ggml_vallex_exp(struct ggml_context *ctx, struct ggml_tensor *a);

struct ggml_tensor *ggml_vallex_sin(struct ggml_context *ctx, struct ggml_tensor *a);

struct ggml_tensor *ggml_vallex_cos(struct ggml_context *ctx, struct ggml_tensor *a);

struct ggml_tensor *ggml_vallex_mul_num(struct ggml_context *ctx, struct ggml_tensor *a, float b);

//https://github.com/pytorch/pytorch/blob/455241bbd3629fb5ff4360c91909c79256935c44/torch/csrc/api/include/torch/nn/functional/linear.h#L19
struct ggml_tensor *ggml_vallex_linear(struct ggml_context *ctx, struct ggml_tensor *input, struct ggml_tensor *weight,
                                       struct ggml_tensor *bias);

struct ggml_tensor *
ggml_vallex_masked_fill(struct ggml_context *ctx, struct ggml_tensor *input, struct ggml_tensor *mask, float value);

// Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
struct ggml_tensor *
ggml_vallex_cat(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b, int dim);

struct ggml_tensor *ggml_vallex_layer_norm(
        struct ggml_context *ctx,
        struct ggml_tensor *input,
        struct ggml_tensor *normalized_shape,
        struct ggml_tensor *weight,
        struct ggml_tensor *bias,
        float eps
);

struct ggml_tensor *ggml_vallex_dropout(struct ggml_context *ctx, struct ggml_tensor *src, float p);

#endif
