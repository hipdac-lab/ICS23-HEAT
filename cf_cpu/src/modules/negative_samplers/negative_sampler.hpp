#pragma once

#include <memory>
#include <vector>
#include "../cf_config.hpp"
#include "../random/uniform.hpp"

namespace cf
{
namespace modules
{
namespace negative_samplers
{

class NegativeSampler
{
  public:
    NegativeSampler(const std::shared_ptr<CFConfig> config, idx_t seed);
    virtual ~NegativeSampler() = default;
    virtual void sampling(std::vector<idx_t>& neg_ids) = 0;
    virtual void ignore_pos_sampling(idx_t user_id, idx_t pos_id, std::vector<idx_t>& neg_ids) = 0;

    idx_t num_negs;
    random::Uniform* neg_sampler;
};

}
}
}