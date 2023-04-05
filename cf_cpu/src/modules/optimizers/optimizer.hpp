#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <iostream>

#include "../cf_config.hpp"
#include "../embeddings/embedding.hpp"

namespace cf
{
namespace modules
{
namespace optimizers
{

class Optimizer
{
  public:
    Optimizer(const std::shared_ptr<CFConfig> config);
    virtual ~Optimizer() = default;
    val_t clip_grad(val_t grad, val_t clip_val);
    void scheduler_step_lr(idx_t epoch, idx_t step_size, val_t gamma);
    void scheduler_multi_step_lr(idx_t epoch, std::vector<idx_t>& milestones, val_t gamma);

    virtual void sparse_step(val_t* emb, val_t* grad) = 0;
    virtual void dense_step(embeddings::Embedding* embedding) = 0;

    idx_t emb_dim;
    val_t clip_val;
    val_t l_r;

    // inline val_t clip_grad(val_t grad, val_t clip_val)
    // {
    //     val_t clipped_grad = std::min(grad, clip_val);
    //     clipped_grad = std::max(clipped_grad, -clip_val);
    //     return clipped_grad;
    // }
};

}
}
}