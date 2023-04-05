#include "model.hpp"

namespace cf
{
namespace modules  
{
namespace models
{

Model::Model(const std::shared_ptr<CFConfig> config, val_t* user_weights, val_t* item_weights)
{
    this->user_embedding = new embeddings::Embedding(config->num_users, config->emb_dim, user_weights);
    this->item_embedding = new embeddings::Embedding(config->num_items, config->emb_dim, item_weights);
}

val_t* Model::read_embedding(embeddings::Embedding* embedding, idx_t idx, val_t* emb_buf)
{
    val_t* emb = embedding->read_weights(idx, emb_buf);
    return emb;
}

void Model::write_embedding(embeddings::Embedding* embedding, idx_t idx, val_t* emb_buf)
{
    embedding->write_weights(idx, emb_buf);
}

val_t* Model::read_gradient(embeddings::Embedding* embedding, idx_t idx, val_t* grad_buf)
{
    val_t* grad = embedding->read_grads(idx, grad_buf);
    return grad;
}

void Model::write_gradient(embeddings::Embedding* embedding, idx_t idx, val_t* grad_buf)
{
    embedding->write_grads(idx, grad_buf);
}

}
}
}