#include "dataset.hpp"

namespace cf
{
namespace modules  
{
namespace datasets
{

Dataset::Dataset(idx_t data_rows, idx_t data_cols, idx_t* vals, idx_t his_items_rows, idx_t his_items_cols, 
    idx_t* his_items_vals, idx_t masks_rows, idx_t masks_cols, idx_t* masks_vals) {
    this->data = new IdxArray(data_rows, data_cols, vals);
    this->data_rows = data_rows;
    this->data_cols = data_cols;

    this->historical_items = new IdxArray(his_items_rows, his_items_cols, his_items_vals);
    this->masks = new IdxArray(masks_rows, masks_cols, masks_vals);
    this->max_his = 0;
}

}
}
}