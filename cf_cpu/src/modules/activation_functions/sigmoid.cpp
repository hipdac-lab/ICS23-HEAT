#include "sigmoid.hpp"

namespace cf
{
namespace modules
{
namespace activation_functions
{

Sigmoid::Sigmoid()
{
}

// std::setprecision(16)
void Sigmoid::activate(val_t input, val_t& output, val_t& grad)
{
    if (input > 0.)
    {
        output = 1. / (1 + std::exp(-input));
    }
    else
    {
        val_t exp_val = std::exp(input);
        output = exp_val / (1. + exp_val);
    }

    val_t grad = output * (1. - output);
}


}
}
}
