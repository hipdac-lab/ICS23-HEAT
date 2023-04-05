#pragma once

#include "negative_sampler.hpp"

namespace cf
{
namespace modules
{
namespace negative_samplers
{

class UniformRandomNegativeSampler : public NegativeSampler
{
  public:
    UniformRandomNegativeSampler(const std::shared_ptr<CFConfig> config, idx_t seed);
    ~UniformRandomNegativeSampler() = default;
    void sampling(std::vector<idx_t>& neg_ids) override;
    void ignore_pos_sampling(idx_t user_id, idx_t pos_id, std::vector<idx_t>& neg_ids) override;
};

}
}
}