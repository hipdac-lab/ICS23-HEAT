#pragma once

#include <vector>
#include "../../splatt/base.h"

namespace cf
{
namespace modules
{

struct CFConfig
{
    CFConfig(idx_t emb_dim, idx_t num_negs, idx_t num_users, idx_t num_items, idx_t train_size, idx_t neg_sampler, idx_t tile_size, idx_t refresh_interval, 
        val_t l2, val_t clip_val, std::vector<idx_t>& milestones, val_t l_r) 
        : emb_dim(emb_dim), num_negs(num_negs), num_users(num_users), num_items(num_items), train_size(train_size), 
            neg_sampler(neg_sampler), tile_size(tile_size), refresh_interval(refresh_interval), l2(l2), clip_val(clip_val), milestones(milestones.begin(), milestones.end()), l_r(l_r)
    {
    }

    idx_t emb_dim;
    idx_t num_negs;
    idx_t num_users;
    idx_t num_items;
    idx_t train_size;
    idx_t neg_sampler;  //0: uniform random 1: random tiling
    idx_t tile_size;
    idx_t refresh_interval;
    val_t l2;
    val_t clip_val;
    std::vector<idx_t> milestones;
    val_t l_r;
};

}
}