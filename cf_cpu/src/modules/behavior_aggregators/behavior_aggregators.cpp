#include "behavior_aggregator.hpp"

namespace cf
{
namespace modules
{
namespace behavior_aggregators
{

// UserHistory::UserHistory(idx_t his_items_rows, idx_t his_items_cols, idx_t* his_items_vals, 
//     idx_t masks_rows, idx_t masks_cols, idx_t* masks_vals)
// {
//     this->historical_items = std::make_shared<memory::Array<idx_t>>(his_items_rows, his_items_cols, his_items_vals);
//     this->masks = std::make_shared<memory::Array<idx_t>>(masks_rows, masks_cols, masks_vals);
//     this->max_his = 0;
// }

AggregatorWeights::AggregatorWeights(int emb_dim, val_t* init_weights0) : emb_dim(emb_dim), weights0(init_weights0, emb_dim, emb_dim)
{
    if (init_weights0 == nullptr)
    {
        this->weights0.setRandom();
    }
}

// aggregation choices: average pooling, self-attention, and user-attention                                             
BehaviorAggregator::BehaviorAggregator(datasets::Dataset* train_data, AggregatorWeights* aggregator_weights, const std::shared_ptr<CFConfig> config) : train_data(train_data), 
    aggregator_weights(aggregator_weights)
{
    this->iteration = 0;
    this->mini_batch_size = 32;
    this->emb_dim = config->emb_dim;
    this->weights0_grad_accu = Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(this->emb_dim, this->emb_dim);
    this->max_his = train_data->max_his;
    this->num_his = this->max_his;
    this->gamma = 0.4;
    this->l_r = config->l_r;

    this->mask_buf = static_cast<idx_t*>(splatt_malloc(sizeof(idx_t))); // Error
    this->his_id_buf = static_cast<idx_t*>(splatt_malloc(this->max_his * sizeof(idx_t)));
    this->omp_means = static_cast<val_t*>(splatt_malloc(this->emb_dim * sizeof(val_t)));
    this->his_emb_buf = static_cast<val_t*>(splatt_malloc(this->emb_dim * sizeof(val_t)));
    this->his_embs = static_cast<val_t*>(splatt_malloc(this->max_his * this->emb_dim * sizeof(val_t)));
    this->his_grad = static_cast<val_t*>(splatt_malloc(this->emb_dim * sizeof(val_t)));
}

// splatt_free(this->his_emb);
// splatt_free(this->his_grad);

void BehaviorAggregator::forward(idx_t user_id, val_t* user_emb_ptr, embeddings::Embedding* item_embedding, memory::ThreadBuffer* t_buf)
{
    double start_time = omp_get_wtime();

    const idx_t emb_dim = this->emb_dim;
    memory::Array<val_t>* item_embedding_weights = item_embedding->weights;
    memory::Array<idx_t>* historical_items = this->train_data->historical_items;
    memory::Array<idx_t>* masks = this->train_data->masks;

    idx_t* his_ids = historical_items->read_row(user_id, this->his_id_buf);
    idx_t* mask = masks->read_row(user_id, this->mask_buf);
    this->num_his = mask[0];
    val_t r_num_his = 1.0 / this->num_his;

    // #pragma omp simd
    // for (idx_t idx0 = 0; idx0 < emb_dim; ++idx0)
    // {
    //     this->omp_means[idx0] = 0;
    // }

    // for (idx_t idx0 = 0; idx0 < this->num_his; ++idx0)
    // {
    //     idx_t his_id = his_ids[idx0];
    //     val_t* his_emb_ptr = item_embedding_weights->read_row(his_id, this->his_emb_buf);

    //     #pragma omp simd
    //     for (idx_t idx1 = 0; idx1 < emb_dim; ++idx1)
    //     {
    //         this->omp_means[idx1] += his_emb_ptr[idx1] * r_num_his;
    //     }
    // }
    // Eigen::Map<Eigen::Array<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> mean_arr(this->omp_means, 1, emb_dim);
    // this->means = mean_arr;

    // #pragma omp master
    // {
    //     std::cout << "iterations ";
    //     for (int i = 0; i < 8; ++i)
    //     {
    //         std::cout << " " << this->omp_means[i];
    //     }
    //     std::cout << std::endl;
    // }

    // Eigen implementation
    for (idx_t idx0 = 0; idx0 < this->num_his; ++idx0)
    {
        idx_t his_id = his_ids[idx0];
        val_t* his_emb_ptr = item_embedding_weights->read_row(his_id, this->his_emb_buf);
        memcpy(this->his_embs + idx0 * emb_dim, his_emb_ptr, emb_dim * sizeof(val_t));
    }

    Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> his_embs_arr(this->his_embs, this->num_his, emb_dim);
    Eigen::Map<Eigen::Array<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> user_emb(user_emb_ptr, 1, emb_dim);
    this->means = his_embs_arr.colwise().sum() * r_num_his;

    // #pragma omp master
    // {
    //     std::cout << "eigen ";
    //     std::cout << this->means;
    //     std::cout << std::endl;
    // }

    double end_time = omp_get_wtime();
    t_buf->time_map["read_his"] = t_buf->time_map["read_his"] + (end_time - start_time);
    start_time = end_time;

    Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic> f_c0 = this->means.matrix() * this->aggregator_weights->weights0.matrix();
    // Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic> outs = this->gamma * user_emb + (1 - this->gamma) * f_c0;
    // memcpy(user_emb_ptr, outs.data(), emb_dim * sizeof(val_t));

    user_emb = this->gamma * user_emb + (1 - this->gamma) * f_c0;

    this->iteration += 1;
    end_time = omp_get_wtime();
    t_buf->time_map["his_mm"] = t_buf->time_map["his_mm"] + (end_time - start_time);
}

void BehaviorAggregator::backward(val_t* outs_grad)
{
    Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> outs_grad_arr(outs_grad, 1, this->emb_dim);
    Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f_c0_grad = outs_grad_arr * (1 - this->gamma);

    Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights0_grad = Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(this->emb_dim, this->emb_dim);
    for (int idx0 = 0; idx0 < this->emb_dim; ++idx0)
    {
        weights0_grad.row(idx0) << this->means(0, idx0) * f_c0_grad;
    }
    this->weights0_grad_accu += weights0_grad;

    if (this->iteration > 0 && (this->iteration % this->mini_batch_size == 0))
    {
        weights0_grad = this->weights0_grad_accu / this->mini_batch_size;
        this->aggregator_weights->weights0 -= this->l_r * weights0_grad;
        this->weights0_grad_accu = Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(this->emb_dim, this->emb_dim);
    }

    #pragma omp simd
    for (idx_t i = 0; i < this->emb_dim; ++i)
    {
        outs_grad[i] *= this->gamma;
    }
}

}
}
}