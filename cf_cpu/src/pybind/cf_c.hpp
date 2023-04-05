#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../splatt/base.h"

using PyMatrix = pybind11::array_t<val_t, pybind11::array::c_style>;

void init_modules(pybind11::module_& cf_module);
