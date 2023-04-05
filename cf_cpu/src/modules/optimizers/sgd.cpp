#include "sgd.hpp"

namespace cf
{
namespace modules
{
namespace optimizers
{

SGD::SGD(const std::shared_ptr<CFConfig> config) : Optimizer(config)
{
}

void SGD::sparse_step(val_t* emb, val_t* grad)
{
    const idx_t emb_dim = this->emb_dim;
    val_t clip_val = this->clip_val;
    val_t l_r = this->l_r;

#pragma omp simd
    for (idx_t i = 0; i < emb_dim; ++i)
    {
        grad[i] = this->clip_grad(grad[i], clip_val);
        emb[i] -= l_r * grad[i];
    }
}

// void SGD::sparse_step(val_t* emb, val_t* grad)
// {
// #pragma omp simd
//     for (idx_t i = 0; i < this->emb_dim; ++i)
//     {
//         grad[i] = this->clip_grad(grad[i], this->clip_val);
//         emb[i] -= this->l_r * grad[i];
//     }
// }

void SGD::dense_step(embeddings::Embedding* embedding)
{
    const idx_t num_embs = embedding->num_embs;
    const idx_t emb_dim = embedding->emb_dim;

#pragma omp parallel
    {
        memory::Array<val_t>* embedding_weights = embedding->weights;
        memory::Array<val_t>* embedding_grads = embedding->grads;
        val_t* emb_buf = new val_t[emb_dim];
        val_t* grad_buf = new val_t[emb_dim];
        val_t* emb_ptr = embedding->weights->data;
        val_t* grad_ptr = embedding->grads->data;

    // #pragma omp for schedule(dynamic, 512)
    #pragma omp for
        for (idx_t i = 0; i < num_embs; ++i)
        {
            val_t* emb_ptr = embedding_weights->read_row(i, emb_buf);
            val_t* grad_ptr = embedding_grads->read_row(i, grad_buf);
            this->sparse_step(emb_ptr, grad_ptr);
            embedding_weights->write_row(i, emb_ptr);
        }
        delete[] emb_buf;
        delete[] grad_buf;
        emb_ptr = nullptr;
        emb_ptr = nullptr;
    }
}

}
}
}



