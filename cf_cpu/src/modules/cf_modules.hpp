#pragma once

#include "cf_config.hpp"
#include "../optimizers/optimizer.hpp"

namespace cf
{
namespace modules
{

struct CFModules
{
    CFModules(CFConfig* cf_config, optimizers::Optimizer* optimizer) : cf_config(cf_config), optimizer(optimizer)
    {
    }

    CFConfig* cf_config;
    optimizers::Optimizer* optimizer;
};

}
}