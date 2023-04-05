#include "click_dataset.hpp"

namespace cf
{
namespace modules  
{  
namespace datasets
{

ClickDataset::ClickDataset(idx_t num_clicks, idx_t click_dim, idx_t* clicks_vals, idx_t his_items_rows, idx_t his_items_cols, 
    idx_t* his_items_vals, idx_t masks_rows, idx_t masks_cols, idx_t* masks_vals) 
    : Dataset(num_clicks, click_dim, clicks_vals, his_items_rows, his_items_cols, his_items_vals, masks_rows, masks_cols, masks_vals)
{
    this->row_buf = static_cast<idx_t*>(splatt_malloc(click_dim * sizeof(idx_t)));
}

void ClickDataset::read_user_item(idx_t idx, idx_t& user_id, idx_t& item_id)
{
    this->data->read_row(idx, this->row_buf);
    user_id = this->row_buf[0];
    item_id = this->row_buf[1];
}

}
}
}
