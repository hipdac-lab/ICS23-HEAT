#pragma once

#include <cmath>
#include "activation_function.hpp"

namespace cf
{
namespace modules
{
namespace activation_functions
{

class Sigmoid : public ActivationFunction
{
  public:
    Sigmoid();
    ~Sigmoid() = default;

    void activate(val_t input, val_t& output, val_t& grad) override;
};

}
}
}