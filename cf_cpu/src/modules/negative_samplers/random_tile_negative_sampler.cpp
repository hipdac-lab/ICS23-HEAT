#include "random_tile_negative_sampler.hpp"

namespace cf
{
namespace modules
{
namespace negative_samplers
{

// neg_tile(config->tile_size, 0)
RandomTileNegativeSampler::RandomTileNegativeSampler(const std::shared_ptr<CFConfig> config, idx_t seed) 
    : NegativeSampler(config, seed), neg_tile(std::vector<idx_t>(config->tile_size))
{
    this->num_negs = config->num_negs;
    this->neg_sampler = new random::Uniform(config->num_items - 1, seed);
    this->tile_sampler = new random::Uniform(config->tile_size - 1, seed);
    this->tile_size = config->tile_size;
    this->refresh_interval = config->refresh_interval;
    this->iterations = 0;
}

void RandomTileNegativeSampler::refresh_tile()
{
    for (idx_t i = 0; i < this->tile_size; ++i)
    {
        idx_t neg_id = this->neg_sampler->read();
        neg_tile[i] = neg_id;
    }
}

void RandomTileNegativeSampler::sampling(std::vector<idx_t>& neg_ids)
{
    if (this->iterations % this->refresh_interval == 0)
    {
        this->refresh_tile();
    }

    for (idx_t i = 0; i < this->num_negs; ++i)
    {
        idx_t tile_idx = this->tile_sampler->read();
        neg_ids[i] = neg_tile[tile_idx];
    }

    this->iterations += 1;
}

void RandomTileNegativeSampler::ignore_pos_sampling(idx_t user_id, idx_t pos_id, std::vector<idx_t>& neg_ids)
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