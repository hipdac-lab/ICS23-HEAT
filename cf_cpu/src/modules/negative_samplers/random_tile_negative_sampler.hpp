#pragma once

#include "negative_sampler.hpp"

namespace cf
{
namespace modules
{
namespace negative_samplers
{

class RandomTileNegativeSampler : public NegativeSampler
{
  public:
    RandomTileNegativeSampler(const std::shared_ptr<CFConfig> config, idx_t seed);
    ~RandomTileNegativeSampler() = default;
    void refresh_tile();
    void sampling(std::vector<idx_t>& neg_ids) override;
    void ignore_pos_sampling(idx_t user_id, idx_t pos_id, std::vector<idx_t>& neg_ids) override;

    idx_t tile_size;
    idx_t refresh_interval;
    idx_t iterations;
    std::vector<idx_t> neg_tile;
    random::Uniform* tile_sampler;
};

}
}
}