#include "engine.hpp"

namespace cf
{
namespace modules
{
namespace train
{

Engine::Engine(std::shared_ptr<datasets::Dataset> train_data,
    std::shared_ptr<behavior_aggregators::AggregatorWeights> aggregator_weights,
    std::shared_ptr<models::Model> model,
    std::shared_ptr<CFConfig> cf_config)
    : train_data(train_data), aggregator_weights(aggregator_weights), model(model), cf_config(cf_config)
{
    this->positive_sampler = std::make_shared<random::Shuffle>(cf_config->train_size);
    optimizers::Optimizer* optimizer = new optimizers::SGD(cf_config);
    this->cf_modules =  std::make_shared<CFModules>(cf_config.get(), optimizer);
    this->random_gen = new random::Uniform(1024, 66);
    this->epoch = 0;
}

void Engine::performance_breakdown(memory::ThreadBuffer* t_buf)
{
    int thread_id = omp_get_thread_num();
    if (thread_id == 0)
    {
        int max_num_threads = omp_get_max_threads();
        std::cout << "thread idx: " << thread_id << " max threads: " << max_num_threads << std::endl;

        double epoch_time = t_buf->time_map["data"] + t_buf->time_map["f_b"];
        double data_p = t_buf->time_map["data"] / epoch_time * 100.0;
        double forward_p = t_buf->time_map["forward"] / epoch_time * 100.0;
        double backward_p = t_buf->time_map["backward"] / epoch_time * 100.0;

        double read_emb_p = t_buf->time_map["read_emb"] / t_buf->time_map["forward"] * 100.0;
        double aggr_f_p = t_buf->time_map["aggr_f"] / t_buf->time_map["forward"] * 100.0;
        double read_his_p = t_buf->time_map["read_his"] / t_buf->time_map["aggr_f"] * 100.0;
        double his_mm_p = t_buf->time_map["his_mm"] / t_buf->time_map["aggr_f"] * 100.0;
        double dot_p = t_buf->time_map["dot"] / t_buf->time_map["forward"] * 100.0;
        double norm_p = t_buf->time_map["norm"] / t_buf->time_map["forward"] * 100.0;
        double loss_p = t_buf->time_map["loss"] / t_buf->time_map["forward"] * 100.0;

        double grad_p = t_buf->time_map["grad"] / t_buf->time_map["backward"] * 100.0;
        double reg_p = t_buf->time_map["reg"] / t_buf->time_map["backward"] * 100.0;
        double write_emb_p = t_buf->time_map["write_emb"] / t_buf->time_map["backward"] * 100.0;
        double aggr_b_p = t_buf->time_map["aggr_b"] / t_buf->time_map["backward"] * 100.0;

        std::cout << " epoch_time: " << epoch_time
        << " data: " << t_buf->time_map["data"] << " data_p %: " << data_p
        << " forward: " << t_buf->time_map["forward"] << " forward_p %: " << forward_p
        << " backward: " << t_buf->time_map["backward"] << " backward_p %: " << backward_p << std::endl
        << " read_emb: " << t_buf->time_map["read_emb"] << " read_emb_p %: " << read_emb_p
        << " aggr_f: " << t_buf->time_map["aggr_f"] << " aggr_f_p %: " << aggr_f_p
        << " read_his: " << t_buf->time_map["read_his"] << " read_his_p %: " << read_his_p
        << " his_mm: " << t_buf->time_map["his_mm"] << " his_mm_p %: " << his_mm_p
        << " dot: " << t_buf->time_map["dot"] << " dot_p %: " << dot_p
        << " norm: " << t_buf->time_map["norm"] << " norm_p %: " << norm_p
        << " loss: " << t_buf->time_map["loss"] << " loss_p %: " << loss_p << std::endl
        << " grad: " << t_buf->time_map["grad"] << " grad_p %: " << grad_p
        << " reg: " << t_buf->time_map["reg"] << " reg_p %: " << reg_p
        << " write_emb: " << t_buf->time_map["write_emb"] << " write_emb_p %: " << write_emb_p
        << " aggr_b: " << t_buf->time_map["aggr_b"] << " aggr_b_p %: " << aggr_b_p
        << std::endl;
        std::cout << std::endl;
    }
}

val_t Engine::train_one_epoch()
{
    auto start = std::chrono::steady_clock::now();

    const idx_t iterations = this->train_data->data_rows;
    idx_t num_negs = this->cf_config->num_negs;
    double loss = 0.;
    idx_t max_threads = omp_get_max_threads();
    datasets::Dataset* train_data_ptr = this->train_data.get();
    // this->positive_sampler->shuffle();
    behavior_aggregators::AggregatorWeights* aggregator_weights_ptr = this->aggregator_weights.get();
    if (this->cf_config->milestones.size() > 1)
    {
        this->cf_modules->optimizer->scheduler_multi_step_lr(this->epoch, this->cf_config->milestones, 0.1);
    }
    else
    {
        this->cf_modules->optimizer->scheduler_step_lr(this->epoch, this->cf_config->milestones[0], 0.1);
    }
    
    Eigen::initParallel();
// #pragma omp parallel reduction(+ : loss) shared(train_data_ptr, aggregator_weights_ptr)
#pragma omp parallel reduction(+ : loss)
    {
        idx_t user_id = 0;
        idx_t pos_id = 0;
        std::vector<idx_t> neg_ids(num_negs);
        idx_t num_threads = omp_get_num_threads();
        idx_t thread_id = omp_get_thread_num();
        // idx_t seed = (this->epoch + 1) * thread_id + this->random_gen->read();
        idx_t seed = (this->epoch + 1) * thread_id;

        negative_samplers::NegativeSampler* negative_sampler = nullptr;
        if (this->cf_config->neg_sampler == 1)
        {
            negative_sampler = static_cast<negative_samplers::NegativeSampler*>(
                new negative_samplers::RandomTileNegativeSampler(this->cf_config, seed));
        }
        else
        {
            negative_sampler = static_cast<negative_samplers::NegativeSampler*>(
                new negative_samplers::UniformRandomNegativeSampler(this->cf_config, seed));
        }
        
        memory::ThreadBuffer* t_buf = new memory::ThreadBuffer(this->cf_config->emb_dim, num_negs);

        // behavior_aggregators::BehaviorAggregator* behavior_aggregator = nullptr;
        behavior_aggregators::BehaviorAggregator behavior_aggregator(train_data_ptr, aggregator_weights_ptr, cf_config);

        #pragma omp master
        {
            std::cout << "iterations " << iterations << " max_threads " << num_threads << std::endl;
        }

// #pragma omp for schedule(dynamic, 4)
// for (idx_t i = 0; i < 16; ++i)
#pragma omp for schedule(dynamic, 512)
        for (idx_t i = 0; i < iterations; ++i)
        {
            double start_time = omp_get_wtime();
            idx_t train_data_idx = this->positive_sampler->read(i);
            this->train_data->read_user_item(train_data_idx, user_id, pos_id);
            negative_sampler->ignore_pos_sampling(user_id, pos_id, neg_ids);
            // negative_sampler->sampling(neg_ids);
            double end_time = omp_get_wtime();
            t_buf->time_map["data"] = t_buf->time_map["data"] + (end_time - start_time);

            loss += this->model->forward_backward(user_id, pos_id, neg_ids, this->cf_modules, t_buf, &behavior_aggregator);
        }
        // performance_breakdown(t_buf);
    }

    // this->cf_modules->optimizer->dense_step(this->model->user_embedding);
    this->model->user_embedding->zero_grad();
    // this->cf_modules->optimizer->dense_step(this->model->item_embedding);
    this->model->item_embedding->zero_grad();

    auto end = std::chrono::steady_clock::now();
    double epoch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1.e-9;
    std::cout << "epoch time: " << epoch_time << " s " << std::endl;

    this->epoch += 1;
    loss /= iterations;
    return loss;
}

void Engine::evaluate0()
{
    idx_t emb_dim = this->cf_config->emb_dim;
    idx_t num_users = this->cf_config->num_users;
    idx_t num_items = this->cf_config->num_items;
    memory::Array<val_t>* user_embedding_weights = this->model->user_embedding->weights;
    memory::Array<val_t>* item_embedding_weights = this->model->item_embedding->weights;
    Eigen::Map<Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> user_emb_mat(user_embedding_weights->data, num_users, emb_dim);
    Eigen::Map<Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> item_emb_mat(item_embedding_weights->data, num_items, emb_dim);
    this->sim_matrix = user_emb_mat * item_emb_mat.transpose();
    // this->sim_matrix = Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic>::Random(100, 100);
}

// PyMatrix Engine::evaluate1()
// {
//     const idx_t iterations = this->test_data->data_rows;
//     idx_t emb_dim = this->cf_config->emb_dim;
//     idx_t num_negs = this->cf_config->num_negs;
//     idx_t num_users = this->cf_config->num_users;
//     idx_t num_items = this->cf_config->num_items;

//     datasets::Dataset* test_data_ptr = this->test_data.get();
//     behavior_aggregators::AggregatorWeights* aggregator_weights_ptr = this->aggregator_weights.get();

//     memory::Array<val_t>* item_embedding_weights = this->model->item_embedding->weights;

//     memory::Array<val_t>* aggr_user_embedding = new ValArray(this->cf_config->num_users, this->cf_config->emb_dim, nullptr);

// // #pragma omp parallel reduction(+ : loss) shared(test_data_ptr, aggregator_weights_ptr)
// #pragma omp parallel reduction(+ : loss)
//     {
//         idx_t user_id = 0;
//         idx_t pos_id = 0;

//         idx_t thread_id = omp_get_thread_num();
//         // idx_t seed = (this->epoch + 1) * thread_id + this->random_gen->read();
//         idx_t seed = (this->epoch + 1) * thread_id;

//         memory::ThreadBuffer* t_buf = new memory::ThreadBuffer(emb_dim, num_negs);

//         // behavior_aggregators::BehaviorAggregator* behavior_aggregator = nullptr;
//         behavior_aggregators::BehaviorAggregator behavior_aggregator(test_data_ptr, aggregator_weights_ptr, cf_config);

// // #pragma omp for schedule(dynamic, 4)
// // for (idx_t i = 0; i < 16; ++i)
// #pragma omp for schedule(dynamic, 512)
//         for (idx_t i = 0; i < iterations; ++i)
//         {
//             double start_time = omp_get_wtime();
//             idx_t test_data_idx = this->positive_sampler->read(i);
//             this->test_data->read_user_item(test_data_idx, user_id, pos_id);
//             memory::Array<val_t>* user_embedding_weights = this->model->user_embedding->weights;
//             val_t* user_emb_ptr = user_embedding_weights->read_row(user_id, t_buf->user_emb_buf);
//             behavior_aggregator.forward(user_id, user_emb_ptr, this->model->item_embedding, t_buf);
//             aggr_user_embedding->write_row(user_id, user_emb_ptr);
//         }
//     }

//     Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> user_emb_mat(aggr_user_embedding->data, num_users, emb_dim);
//     Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> item_emb_mat(item_embedding_weights->data, num_items, emb_dim);
//     this->sim_matrix = user_emb_mat * item_emb_mat.transpose();
//     return PyMatrix({num_users, num_items}, {num_items * sizeof(val_t), sizeof(val_t)}, this->sim_matrix.data());
// }


}
}
}

// rm -rf ./cf_c.cpython-38-x86_64-linux-gnu.so && make -j && cp ./cf_c.cpython-38-x86_64-linux-gnu.so ../cf/ 