#include "cf_c.hpp"

#include "../modules/cf_config.hpp"
#include "../modules/datasets/click_dataset.hpp"
#include "../modules/models/matrix_factorization.hpp"
#include "../modules/behavior_aggregators/behavior_aggregator.hpp"
#include "../modules/train/engine.hpp"

namespace py = pybind11;

void init_cf_config(py::module_& modules_module)
{
    py::class_<cf::modules::CFConfig, 
        std::shared_ptr<cf::modules::CFConfig>>(modules_module, "CFConfig")
        .def(py::init<idx_t, idx_t, idx_t, idx_t, idx_t, idx_t, idx_t, idx_t, val_t, val_t, std::vector<idx_t>&, val_t>(),
            py::arg("emb_dim"),
            py::arg("num_negs"),
            py::arg("num_users"),
            py::arg("num_items"),
            py::arg("train_size"),
            py::arg("neg_sampler"),
            py::arg("tile_size"),
            py::arg("refresh_interval"),
            py::arg("l2"),
            py::arg("clip_val"),
            py::arg("milestones"),
            py::arg("l_r"))
        
        .def_readwrite("emb_dim", &cf::modules::CFConfig::emb_dim);
}

void init_datasets(py::module_& modules_module)
{
    py::module_ datasets_module = modules_module.def_submodule("datasets", "datasets");

    py::class_<cf::modules::datasets::Dataset, 
        std::shared_ptr<cf::modules::datasets::Dataset>>(datasets_module, "Dataset");

    py::class_<cf::modules::datasets::ClickDataset, 
        cf::modules::datasets::Dataset, 
        std::shared_ptr<cf::modules::datasets::ClickDataset>>(datasets_module, "ClickDataset")
        .def(py::init<>([](py::array_t<idx_t, py::array::c_style>& click_dataset, py::array_t<idx_t, py::array::c_style>& historical_items, py::array_t<idx_t, py::array::c_style>& masks) { 
            idx_t num_clicks = click_dataset.shape()[0];
            idx_t click_dim = click_dataset.shape()[1];
            auto clicks_vals = static_cast<idx_t*>(click_dataset.request().ptr);
            idx_t his_items_rows = historical_items.shape()[0];
            idx_t his_items_cols = historical_items.shape()[1];
            auto his_items_vals = static_cast<idx_t*>(historical_items.request().ptr);
            idx_t masks_rows = masks.shape()[0];
            idx_t masks_cols = masks.shape()[1];
            auto masks_vals = static_cast<idx_t*>(masks.request().ptr);
            return cf::modules::datasets::ClickDataset(num_clicks, click_dim, clicks_vals, his_items_rows, 
                his_items_cols, his_items_vals, masks_rows, masks_cols, masks_vals); }),
            py::arg("click_dataset"),
            py::arg("historical_items"),
            py::arg("masks"))
        .def_readwrite("data_rows", &cf::modules::datasets::ClickDataset::data_rows)
        .def_readwrite("max_his", &cf::modules::datasets::ClickDataset::max_his);
}

void init_models(py::module_& modules_module)
{
    py::module_ models_module = modules_module.def_submodule("models", "models");

    py::class_<cf::modules::models::Model,
        std::shared_ptr<cf::modules::models::Model>>(models_module, "Model");

    py::class_<cf::modules::models::MatrixFactorization,
        cf::modules::models::Model,
        std::shared_ptr<cf::modules::models::MatrixFactorization>>(models_module, "MatrixFactorization")
        .def(py::init<>([](const std::shared_ptr<cf::modules::CFConfig> cf_config, 
            py::array_t<val_t, py::array::c_style>& user_weights,
            py::array_t<val_t, py::array::c_style>& item_weights) { 
                //  idx_t user_weights_rows = click_dataset.shape()[0];
                //  idx_t user_weights_cols = click_dataset.shape()[1];
                auto user_weights_ptr = static_cast<val_t*>(user_weights.request().ptr);
                auto item_weights_ptr = static_cast<val_t*>(item_weights.request().ptr);
                return cf::modules::models::MatrixFactorization(cf_config, user_weights_ptr, item_weights_ptr); }),
            py::arg("cf_config"),
            py::arg("user_weights"),
            py::arg("item_weights"));

        // .def_readwrite("var", &cf::modules::models::MatrixFactorization::var);
}

void init_behavior_aggregators(py::module_& modules_module)
{
    py::module_ behavior_aggregators_module = modules_module.def_submodule("behavior_aggregators", "Behavior aggregators");

    py::class_<cf::modules::behavior_aggregators::AggregatorWeights, 
        std::shared_ptr<cf::modules::behavior_aggregators::AggregatorWeights>>(behavior_aggregators_module, "AggregatorWeights")
        .def(py::init<>([](py::array_t<val_t, py::array::c_style>& aggregator_weights0) { 
            idx_t emb_dim = aggregator_weights0.shape()[0];
            auto init_weights0 = static_cast<val_t*>(aggregator_weights0.request().ptr);
            return cf::modules::behavior_aggregators::AggregatorWeights(emb_dim, init_weights0); }),
            py::arg("aggregator_weights0"))
        .def_readwrite("emb_dim", &cf::modules::behavior_aggregators::AggregatorWeights::emb_dim);
}

void init_train(py::module_& modules_module)
{
    py::module_ train_module = modules_module.def_submodule("train", "train");

    py::class_<cf::modules::train::Engine, 
        std::shared_ptr<cf::modules::train::Engine>>(train_module, "Engine")
        .def(py::init<std::shared_ptr<cf::modules::datasets::Dataset>, 
            std::shared_ptr<cf::modules::behavior_aggregators::AggregatorWeights>, 
            std::shared_ptr<cf::modules::models::Model>, 
            std::shared_ptr<cf::modules::CFConfig>>(),
            py::arg("dataset"),
            py::arg("aggregator_weights"),
            py::arg("model"),
            py::arg("cf_config"))

        .def("train_one_epoch", &cf::modules::train::Engine::train_one_epoch)

        // .def("evaluate0", &cf::modules::train::Engine::evaluate0);
        
        .def("evaluate0",
            [](cf::modules::train::Engine& engine) {
                engine.evaluate0();
                size_t num_rows = engine.sim_matrix.rows();
                size_t num_cols = engine.sim_matrix.cols();
                val_t* sim_mat_ptr = engine.sim_matrix.data();
                return PyMatrix({num_rows, num_cols}, {num_cols * sizeof(val_t), sizeof(val_t)}, sim_mat_ptr);
             });
        // .def("read_sim_matrix", &cf::modules::train::Engine::read_sim_matrix);
        // .def_readwrite("var", &cf::modules::train::Engine::var);
}

void init_modules(py::module_& cf_module)
{
    py::module_ modules_module = cf_module.def_submodule("modules", "modules");

    init_cf_config(modules_module);
    init_datasets(modules_module);
    init_models(modules_module);
    init_behavior_aggregators(modules_module);
    init_train(modules_module);
}
