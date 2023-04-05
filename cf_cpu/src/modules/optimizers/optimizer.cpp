#include "optimizer.hpp"

namespace cf
{
namespace modules
{
namespace optimizers
{

Optimizer::Optimizer(const std::shared_ptr<CFConfig> config)
{
    this->emb_dim = config->emb_dim;
    this->clip_val = config->clip_val;
    this->l_r = config->l_r;
}

val_t Optimizer::clip_grad(val_t grad, val_t clip_val)
{
    val_t clipped_grad = std::min(grad, clip_val);
    clipped_grad = std::max(clipped_grad, -clip_val);
    return clipped_grad;
}

void Optimizer::scheduler_step_lr(idx_t epoch, idx_t step_size, val_t gamma)
{
    if (epoch > 0 && epoch % step_size == 0)
    {
        this->l_r = this->l_r * gamma;
    }
}

void Optimizer::scheduler_multi_step_lr(idx_t epoch, std::vector<idx_t>& milestones, val_t gamma)
{
    if (std::find(milestones.begin(), milestones.end(), epoch) != milestones.end())
    {
        this->l_r = this->l_r * gamma;
    }
}

}
}
}