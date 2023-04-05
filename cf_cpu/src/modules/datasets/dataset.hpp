#pragma once

#include "../memory/array.hpp"

namespace cf
{
namespace modules  
{
namespace datasets
{

using IdxArray = memory::Array<idx_t>;
// typedef memory::Array<idx_t> IdxArray;

class Dataset
{
  public:
    Dataset(idx_t data_rows, idx_t data_cols, idx_t* vals, idx_t his_items_rows, idx_t his_items_cols, idx_t* his_items_vals, 
        idx_t masks_rows, idx_t masks_cols, idx_t* masks_vals);
    virtual ~Dataset() = default;
    virtual void read_user_item(idx_t idx, idx_t& user_id, idx_t& item_id) = 0;

    IdxArray* data;
    idx_t data_rows;
    idx_t data_cols;

    IdxArray* historical_items;
    IdxArray* masks;
    int max_his;
};

}
}
}