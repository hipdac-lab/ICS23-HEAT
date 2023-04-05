#pragma once

#include <memory>
#include <iostream>
#include <Eigen/Dense>

#include "../../splatt/base.h"
#include "../memory/array.hpp"
#include "../embeddings/embedding.hpp"
#include "../datasets/dataset.hpp"
#include "../memory/thread_buffer.hpp"
#include "../cf_config.hpp"

namespace cf
{
namespace modules
{
namespace behavior_aggregators
{

// class UserHistory
// {
//   public:
//     UserHistory(idx_t his_items_rows, idx_t his_items_cols, idx_t* his_items_vals, 
//         idx_t masks_rows, idx_t masks_cols, idx_t* masks_vals);
//     ~UserHistory() = default;

//     std::shared_ptr<memory::Array<idx_t>> historical_items;
//     std::shared_ptr<memory::Array<idx_t>> masks;
//     int max_his;

//     // memory::Array<idx_t>* historical_items;
//     // memory::Array<idx_t>* masks;
// };

class AggregatorWeights
{
  public:
    AggregatorWeights(int emb_dim, val_t* init_weights0);
    ~AggregatorWeights() = default;
    int emb_dim;
    // Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights0;
    // Eigen::Map<Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weights0;
    Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weights0;
};

class BehaviorAggregator
{
  public:
    BehaviorAggregator(datasets::Dataset* train_data, AggregatorWeights* aggregator_weights, const std::shared_ptr<CFConfig> config);
    ~BehaviorAggregator() = default;
    void forward(idx_t user_id, val_t* user_emb_ptr, embeddings::Embedding* item_embedding, memory::ThreadBuffer* t_buf);
    void backward(val_t* outs_grad);
    
    datasets::Dataset* train_data;
    AggregatorWeights* aggregator_weights;
    Eigen::Array<val_t, 1, Eigen::Dynamic, Eigen::RowMajor> means;
    Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights0_grad_accu;

    idx_t iteration;
    idx_t mini_batch_size; 
    int emb_dim;
    int max_his;
    int num_his;
    val_t gamma;
    val_t l_r;

    idx_t* mask_buf; 
    idx_t* his_id_buf;
    val_t* omp_means;
    val_t* his_emb_buf;
    val_t* his_embs;
    val_t* his_grad;
};

}
}
}