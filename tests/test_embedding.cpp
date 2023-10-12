#include "embedding.h"
#include "gtest/gtest.h"

#include <vector>
#include <cstdio>

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

// TokenEmbedding INIT
TEST(TokenEmbedding_Init, BasicAssertions) {
//    const auto tk = new TokenEmbedding(1, 2, 3);
//    tk->set_data(ggml_new_tensor_1d(nullptr, GGML_TYPE_Q5_0, 1));
//    auto data = tk->forward(ggml_new_tensor_1d(nullptr, GGML_TYPE_Q5_0, 1))->data;
//    int result = reinterpret_cast<int>(data);
    EXPECT_EQ(42, 42);
}

TEST(MINI_GGML_STATR,ADD){
    const struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 10240,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true
    };
    // init ggml_time total
    ggml_time_init();
    const auto t_main_start_us = ggml_time_us();
    // create ggml ctx
    const auto ctx = ggml_init(params);
    // create ggml cpu backend
    const auto backend = ggml_backend_cpu_init();
    // create 1d tensor
    const auto tensor_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    const auto tensor_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto tensor_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    std::vector<float> tensor_data_a = {1.0};
    std::vector<float> tensor_data_b = {2.0};
    std::vector<float> tensor_data_c;

    const auto tensor_1d_memory_size = ggml_nbytes(tensor_a) + ggml_nbytes(tensor_b);
    // create a backend buffer (can be in host or device memory)
    const auto tensor_1d_buffer = ggml_backend_alloc_buffer(backend, tensor_1d_memory_size + 256);

    // set value
    {
        const auto alloc = ggml_allocr_new_from_buffer(tensor_1d_buffer);
        // this updates the pointers in the tensors to point to the correct location in the buffer
        // this is necessary since the ggml_context is .no_alloc == true
        // note that the buffer can actually be a device buffer, depending on the backend
        ggml_allocr_alloc(alloc, tensor_a);
        ggml_allocr_alloc(alloc, tensor_b);

        // in cpu we also can do
        // tensor_a->data = &tensor_data_a.data();
        // tensor_b->data = &tensor_data_b.data();
        ggml_backend_tensor_set(tensor_a, tensor_data_a.data(), 0, ggml_nbytes(tensor_a));
        ggml_backend_tensor_set(tensor_b, tensor_data_b.data(), 0, ggml_nbytes(tensor_b));
        ggml_allocr_free(alloc);
    }

    // compute
    {
        const auto compute_tensor_buffer = ggml_backend_alloc_buffer(backend, 656480);
        const auto allocr = ggml_allocr_new_from_buffer(compute_tensor_buffer);
        const auto gf = ggml_new_graph(ctx);

        // creat forward
        tensor_c = ggml_add(ctx, tensor_a, tensor_b);
        ggml_build_forward_expand(gf, tensor_c);

        // allocate tensors
        ggml_allocr_alloc_graph(allocr, gf);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, 1);
        }

        ggml_backend_graph_compute(backend, gf);

        tensor_data_c.resize(1);
        ggml_backend_tensor_get(gf->nodes[gf->n_nodes - 1], tensor_data_c.data(), 0,
                                tensor_data_c.size() * sizeof(float));
        printf("result is = [");
        for (auto i = tensor_data_c.begin(); i != tensor_data_c.end(); ++i) {
            printf("%f ", *i);
        }
        printf("]\n");
        ggml_allocr_free(allocr);
    }

    const auto t_main_end_us = ggml_time_us();

    printf("total time = %8.2f ms\n", (t_main_end_us - t_main_start_us) / 1000.0f);

    ggml_free(ctx);
    ggml_backend_buffer_free(tensor_1d_buffer);
    ggml_backend_free(backend);
}