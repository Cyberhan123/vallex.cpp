
#pragma once
#ifndef EMBEDDING_H
#define EMBEDDING_H
class SinePositionalEmbedding
{
public:
    SinePositionalEmbedding(int d_model, float dropout, bool scale, bool normalize);
};

class TokenEmbedding
{
public:
    TokenEmbedding(int d_model, int num_token);
};
#endif