#pragma once

#include <omp.h>
#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "../cf_modules.hpp"
#include "../cf_config.hpp"
#include "../embeddings/embedding.hpp"
#include "../memory/thread_buffer.hpp"
#include "../behavior_aggregators/behavior_aggregator.hpp"

namespace cf
{
namespace modules  
{
namespace models
{

class Model
{
  public:
    Model(const std::shared_ptr<CFConfig> config, val_t* user_weights, val_t* item_weights);
    virtual ~Model() = default;
    val_t* read_embedding(embeddings::Embedding* embedding, idx_t idx, val_t* emb_buf);
    void write_embedding(embeddings::Embedding* embedding, idx_t idx, val_t* emb_buf);
    val_t* read_gradient(embeddings::Embedding* embedding, idx_t idx, val_t* grad_buf);
    void write_gradient(embeddings::Embedding* embedding, idx_t idx, val_t* grad_buf);

    virtual val_t forward_backward(idx_t user_id, idx_t pos_id, std::vector<idx_t>& neg_ids, 
        const std::shared_ptr<CFModules> cf_modules, memory::ThreadBuffer* t_buf, behavior_aggregators::BehaviorAggregator* behavior_aggregator) = 0;

    embeddings::Embedding* user_embedding;
    embeddings::Embedding* item_embedding;
};

}
}
}