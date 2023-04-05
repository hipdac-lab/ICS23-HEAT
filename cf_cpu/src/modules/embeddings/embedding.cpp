#include "embedding.hpp"

namespace cf
{
namespace modules  
{
namespace embeddings
{

Embedding::Embedding(idx_t num_embs, idx_t emb_dim, val_t* init_weights)
{
    this->weights = new ValArray(num_embs, emb_dim, init_weights);
    this->grads = new ValArray(num_embs, emb_dim, nullptr);

    this->num_embs = num_embs;
    this->emb_dim = emb_dim;
}

val_t* Embedding::read_weights(idx_t idx, val_t* weight_buf)
{
    std::memcpy(weight_buf, this->weights->data + (idx * this->emb_dim), this->emb_dim * sizeof(val_t));
    return weight_buf;
}

void Embedding::write_weights(idx_t idx, val_t* weight_buf)
{
    std::memcpy(this->weights->data + (idx * this->emb_dim), weight_buf, this->emb_dim * sizeof(val_t));
}

val_t* Embedding::read_grads(idx_t idx, val_t* grad_buf)
{
    std::memcpy(grad_buf, this->grads->data + (idx * this->emb_dim), this->emb_dim * sizeof(val_t));
    return grad_buf;
}

void Embedding::write_grads(idx_t idx, val_t* grad_buf)
{
    std::memcpy(this->grads->data + (idx * this->emb_dim), grad_buf, this->emb_dim * sizeof(val_t));
}

void Embedding::zero_grad()
{
    idx_t num_bytes = this->num_embs * this->emb_dim * sizeof(val_t);
    par_memset(this->grads->data, 0, num_bytes);
}

}
}
}

