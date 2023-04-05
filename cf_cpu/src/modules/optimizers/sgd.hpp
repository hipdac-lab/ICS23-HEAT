#pragma once

#include "optimizer.hpp"

namespace cf
{
namespace modules
{
namespace optimizers
{

class SGD : public Optimizer
{
  public:
    SGD(const std::shared_ptr<CFConfig> config);
    ~SGD() = default;

    void sparse_step(val_t* emb, val_t* grad) override;
    void dense_step(embeddings::Embedding* embedding) override;
};

}
}
}