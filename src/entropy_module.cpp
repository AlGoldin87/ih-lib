#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "entropy_core.hpp"

namespace py = pybind11;

double calculate_entropy_py(py::array_t<int> data, py::array_t<int> mask) {
    if (data.ndim() != 2) throw std::invalid_argument("Data must be 2D");
    if (mask.ndim() != 1) throw std::invalid_argument("Mask must be 1D");
    
    auto data_buf = data.request();
    auto mask_buf = mask.request();
    
    size_t n_rows = data_buf.shape[0];
    size_t n_cols = data_buf.shape[1];
    size_t mask_size = mask_buf.shape[0];
    
    if (n_cols != mask_size) {
        throw std::invalid_argument("Number of columns doesn't match mask size");
    }
    
    int* data_ptr = static_cast<int*>(data_buf.ptr);
    int* mask_ptr = static_cast<int*>(mask_buf.ptr);
    
    return calculate_entropy(data_ptr, n_rows, n_cols, mask_ptr, mask_size);
}

// Модуль называется _core (будет IH._core)
PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for IH library ";
    m.def("calculate_entropy", &calculate_entropy_py,
          py::arg("data"), py::arg("mask"),
          "Calculate entropy for selected features");
}