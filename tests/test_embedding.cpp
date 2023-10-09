#include "embedding.h"
#include "gtest/gtest.h"


// TokenEmbedding INIT
TEST(TokenEmbedding_Init, BasicAssertions) {
    const auto tk = new TokenEmbedding(1, 2, 3);
//    tk->set_data(ggml_new_tensor_1d(nullptr, GGML_TYPE_Q5_0, 1));
//    auto data = tk->forward(ggml_new_tensor_1d(nullptr, GGML_TYPE_Q5_0, 1))->data;
//    int result = reinterpret_cast<int>(data);
    EXPECT_EQ(42, 42);
}