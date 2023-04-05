#pragma once

#include <algorithm>
#include "../../splatt/base.h"

namespace cf
{
namespace modules
{
namespace random
{

class Shuffle
{
    public:
        Shuffle(idx_t idx_size)
        {
            this->idx_size = idx_size;
            this->indices = static_cast<idx_t*>(splatt_malloc(idx_size * sizeof(idx_t)));
            for (idx_t i = 0; i < idx_size; ++i)
            {
                this->indices[i] = i;
            }
            // this->shuffle();
        }

        ~Shuffle()
        {
            if (this->indices != nullptr)
            {
                splatt_free(this->indices);
                this->indices = nullptr;
            }
        }

        void shuffle()
        {
            std::random_shuffle(this->indices, &(this->indices[this->idx_size - 1]));
        }

        idx_t read(idx_t idx)
        {
            return this->indices[idx];
        }

        idx_t idx_size;
        idx_t* indices;
};

}
}
}