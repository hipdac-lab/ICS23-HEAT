#pragma once

#include <random>
#include "../../splatt/base.h"

namespace cf
{
namespace modules
{
namespace random
{

class Uniform
{
    public:
        Uniform(idx_t max_idx, idx_t seed)
        {
            this->max_idx = max_idx;
            this->rng.seed(seed);
            this->dist = std::uniform_int_distribution<std::mt19937_64::result_type>(0, max_idx);
        }

        ~Uniform()
        {
        }

        idx_t read()
        {
            return this->dist(rng);
        }

        idx_t max_idx;
        std::mt19937_64 rng;
        std::uniform_int_distribution<std::mt19937_64::result_type> dist;
};

}
}
}
