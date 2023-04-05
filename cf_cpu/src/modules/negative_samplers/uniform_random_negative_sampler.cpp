#include "uniform_random_negative_sampler.hpp"

namespace cf
{
namespace modules
{
namespace negative_samplers
{

UniformRandomNegativeSampler::UniformRandomNegativeSampler(const std::shared_ptr<CFConfig> config, idx_t seed) 
    : NegativeSampler(config, seed)
{
    this->num_negs = config->num_negs;
    this->neg_sampler = new random::Uniform(config->num_items - 1, seed);
}

void UniformRandomNegativeSampler::sampling(std::vector<idx_t>& neg_ids)
{
    for (idx_t i = 0; i < this->num_negs; ++i)
    {
        idx_t neg_id = this->neg_sampler->read();
        neg_ids[i] = neg_id;
    }
}

void UniformRandomNegativeSampler::ignore_pos_sampling(idx_t user_id, idx_t pos_id, std::vector<idx_t>& neg_ids)
{
    for (idx_t i = 0; i < this->num_negs; ++i)
    {
        idx_t neg_id = this->neg_sampler->read();
        if (neg_id != pos_id)
        {
            neg_ids[i] = neg_id;
        }
    }
}

}
}
}