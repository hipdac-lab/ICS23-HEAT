#pragma once

#include "../../splatt/base.h"

namespace cf
{
namespace modules
{
namespace activation_functions
{

class ActivationFunction
{
  public:
    ActivationFunction();
    virtual ~ActivationFunction() = default;
    virtual void activate(val_t input, val_t& output, val_t& grad);
};

}
}
}