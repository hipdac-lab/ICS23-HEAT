
#pragma once

#include "dataset.hpp"
#include "../memory/array.hpp"

namespace cf
{
namespace modules  
{  
namespace datasets
{

class ClickDataset : public Dataset
{
  public:
    ClickDataset(idx_t num_clicks, idx_t click_dim, idx_t* clicks_vals, idx_t his_items_rows, idx_t his_items_cols, 
        idx_t* his_items_vals, idx_t masks_rows, idx_t masks_cols, idx_t* masks_vals);
    ~ClickDataset() = default;
    void read_user_item(idx_t idx, idx_t& user_id, idx_t& item_id);

    idx_t* row_buf;
};

}
}
}