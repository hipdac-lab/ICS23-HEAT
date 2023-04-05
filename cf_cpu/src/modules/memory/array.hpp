#pragma once

#include "../../splatt/base.h"
#include "../../splatt/util.h"

namespace cf
{
namespace modules
{
namespace memory
{

template<typename t> 
class Array
{
public:
    Array(idx_t num_rows, idx_t num_cols, t* vals) : num_rows(num_rows), num_cols(num_cols)
    {
        if (vals == nullptr)
        {
            idx_t num_bytes = num_rows * num_cols * sizeof(t);
            this->data = static_cast<t*>(splatt_malloc(num_bytes));
            par_memset(this->data, 0, num_bytes);
            this->free_data = true;

            // this->data = new val_t[num_embs * emb_dim];
            // par_memset(this->data, 0, num_bytes);
            // this->delete_data = true;
        }
        else
        {
            this->data = vals;
            this->free_data = false;
        }
    }

    ~Array()
    {
        if (this->data != nullptr && this->free_data)
        {
            splatt_free(this->data);
            this->data = nullptr;
        }
    }

    t* read_row(idx_t row, t* vals)
    {
        memcpy(vals, this->data + (row * this->num_cols), this->num_cols * sizeof(t));
        return vals;
    }

    void write_row(idx_t row, t* vals)
    {
        memcpy(this->data + (row * this->num_cols), vals, this->num_cols * sizeof(t));
    }

    idx_t num_rows;
    idx_t num_cols;
    t* data;
    bool free_data;
};

}
}
}
