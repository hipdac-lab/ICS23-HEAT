#pragma once

#include <omp.h>
#include <iostream>
#include <chrono>

#include "../datasets/click_dataset.hpp"
#include "../behavior_aggregators/behavior_aggregator.hpp"
#include "../models/model.hpp"
#include "../negative_samplers/uniform_random_negative_sampler.hpp"
#include "../negative_samplers/random_tile_negative_sampler.hpp"

#include "../random/shuffle.hpp"
#include "../optimizers/sgd.hpp"
#include "../cf_modules.hpp"


namespace cf
{
namespace modules
{
namespace train
{

class Engine
{
  public:
    Engine(std::shared_ptr<datasets::Dataset> train_data,
        std::shared_ptr<behavior_aggregators::AggregatorWeights> aggregator_weights,
        std::shared_ptr<models::Model> model,
        std::shared_ptr<CFConfig> cf_config);

    ~Engine() = default;
    val_t train_one_epoch();
    void performance_breakdown(memory::ThreadBuffer* t_buf);
    void evaluate0();
    // void evaluate1();

    std::shared_ptr<datasets::Dataset> train_data;
    // std::shared_ptr<datasets::Dataset> test_data;
    std::shared_ptr<behavior_aggregators::AggregatorWeights> aggregator_weights;
    std::shared_ptr<models::Model> model;
    std::shared_ptr<random::Shuffle> positive_sampler;
    std::shared_ptr<CFConfig> cf_config;
    std::shared_ptr<CFModules> cf_modules;

    random::Uniform* random_gen;
    idx_t epoch;

    Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sim_matrix;
};

}
}
}