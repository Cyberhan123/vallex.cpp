#pragma once
#ifndef COMMON_H
#define COMMON_H

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#include <string>
#include <map>


enum LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

static auto log_level = LogLevel::INFO;

#ifndef FILENAME__
#define FILENAME__ "vallex.cpp"
#endif

#define LOGGER(level, format, ...)                                                                    \
    do {                                                                                              \
        if (level < log_level) {                                                                      \
            break;                                                                                    \
        }                                                                                             \
        if (level == SDLogLevel::DEBUG) {                                                             \
            printf("[DEBUG] %s:%-4d - " format "\n", FILENAME__, __LINE__, ##__VA_ARGS__);          \
            fflush(stdout);                                                                           \
        } else if (level == SDLogLevel::INFO) {                                                       \
            printf("[INFO]  %s:%-4d - " format "\n", FILENAME__, __LINE__, ##__VA_ARGS__);          \
            fflush(stdout);                                                                           \
        } else if (level == SDLogLevel::WARN) {                                                       \
            fprintf(stderr, "[WARN]  %s:%-4d - " format "\n", FILENAME__, __LINE__, ##__VA_ARGS__); \
            fflush(stdout);                                                                           \
        } else if (level == SDLogLevel::ERROR) {                                                      \
            fprintf(stderr, "[ERROR] %s:%-4d - " format "\n", FILENAME__, __LINE__, ##__VA_ARGS__); \
            fflush(stdout);                                                                           \
        }                                                                                             \
    } while (0)

#define LOG_DEBUG(format, ...) LOGGER(LogLevel::DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) LOGGER(LogLevel::INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) LOGGER(LogLevel::WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) LOGGER(LogLevel::ERROR, format, ##__VA_ARGS__)


namespace NN {
    template<typename... Ts>
    class Module {
        // compute module buff size
        virtual size_t compute_params_mem_size(ggml_type wtype) = 0;

        // init tensor struct
        virtual void init_params(struct ggml_context *ctx, ggml_type wtype) = 0;

        // mapping tensor struct to buff
        virtual void mapping_tensor(std::map<std::string, struct ggml_tensor *> &tensors, std::string prefix) = 0;

        virtual struct ggml_tensor *forward(struct ggml_graph_ctx *ctx, Ts... Args) = 0;
    };
}

std::string get_system_info();


//class LifeCycle {
//public:
//    explicit LifeCycle(ggml_init_params params);
//
//    bool to_gpu();
//
//    template<typename... Ts>  bool register_as(NN::Module<Ts> module, std::string name);
//
//    bool to_cpu();
//
//protected:
//    struct ggml_backend *backend;
//    struct ggml_context *ctx;
//    struct ggml_allocr *allocr;
//    int64_t t_start_us;
//};

struct ggml_graph_ctx {
    struct ggml_context *ctx;
    struct ggml_backend *backend;
    struct ggml_allocr *allocr;
};

struct ggml_tensor *ggml_arange_1d(struct ggml_graph_ctx *ctx, enum ggml_type type, int start, int end,int step);

struct ggml_tensor *ggml_arange_2d(struct ggml_graph_ctx *ctx, enum ggml_type type, int start, int end,int step);

struct ggml_tensor *ggml_exp_1d(struct ggml_graph_ctx *ctx)

#endif
