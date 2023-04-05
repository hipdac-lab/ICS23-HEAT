
#pragma once

#include <cstring>
#include "../memory/array.hpp"

namespace cf
{
namespace modules  
{
namespace embeddings
{

using ValArray = memory::Array<val_t>;
// typedef memory::Array<val_t> ValArray;

class Embedding
{
  public:
    Embedding(idx_t num_embs, idx_t emb_dim, val_t* init_weights);
    ~Embedding() = default;
    val_t* read_weights(idx_t idx, val_t* weight_buf);
    void write_weights(idx_t idx, val_t* weight_buf);
    val_t* read_grads(idx_t idx, val_t* grad_buf);
    void write_grads(idx_t idx, val_t* grad_buf);
    void zero_grad();

    idx_t num_embs;
    idx_t emb_dim;
    ValArray* weights;
    ValArray* grads;
};

}
}
}