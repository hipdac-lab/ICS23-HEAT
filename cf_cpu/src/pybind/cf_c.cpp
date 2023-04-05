
#include "cf_c.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cf_c, cf_module) {
    cf_module.doc() = "CF_CPU pybind11";
    init_modules(cf_module);
}