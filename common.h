#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <ggml/ggml.h>
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

        virtual size_t compute_params_mem_size(ggml_type wtype) = 0;

        virtual void init_params(struct ggml_context *ctx, ggml_type wtype) = 0;

        virtual void mapping_tensor(std::map<std::string, struct ggml_tensor*>& tensors, std::string prefix)=0;

        virtual struct ggml_tensor *forward(struct ggml_context *ctx, Ts... Args) = 0;
    };
}

std::string get_system_info();

#endif
