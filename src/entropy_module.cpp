#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "entropy_core.hpp"
#include "../include/rule_finder.hpp"

namespace py = pybind11;

// ==================== СУЩЕСТВУЮЩАЯ ФУНКЦИЯ ====================
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

// ==================== НОВАЯ ФУНКЦИЯ ====================
py::list find_best_rules_py(
    py::array_t<float> raw_data,
    py::list feature_mask_py,
    int y_index,
    double y_class_low,
    double y_class_high,
    py::list sharpness_list_py,
    py::list feature_names_py) {
    
    auto buf = raw_data.request();
    if (buf.ndim != 2) throw std::invalid_argument("raw_data must be 2D");
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    float* ptr = static_cast<float*>(buf.ptr);
    
    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i][j] = ptr[i * cols + j];
        }
    }
    
    std::vector<int> feature_mask;
    for (auto item : feature_mask_py) {
        feature_mask.push_back(item.cast<int>());
    }
    
    std::vector<double> sharpness_list;
    for (auto item : sharpness_list_py) {
        sharpness_list.push_back(item.cast<double>());
    }
    
    std::vector<std::string> feature_names;
    for (auto item : feature_names_py) {
        feature_names.push_back(item.cast<std::string>());
    }
    
    auto results = find_best_rules(
        data, feature_mask, y_index, 
        y_class_low, y_class_high, 
        sharpness_list, feature_names
    );
    
    py::list py_results;
    for (const auto& res : results) {
        py::dict d;
        d["Rxy"] = res.Rxy;
        d["L"] = res.L;
        d["R"] = res.R;
        d["inverted"] = res.inverted;
        d["Hx"] = res.Hx;
        d["Hy"] = res.Hy;
        d["Ixy"] = res.Ixy;
        d["class0"] = res.class0_count;
        d["class1"] = res.class1_count;
        d["interval_start"] = res.interval_start;
        d["interval_end"] = res.interval_end;
        d["feature_name"] = res.feature_name;
        d["rule_text"] = res.rule_text;
        py_results.append(d);
    }
    
    return py_results;
}

// ==================== МОДУЛЬ ====================
PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for IH library";
    
    m.def("calculate_entropy", &calculate_entropy_py,
          py::arg("data"), py::arg("mask"),
          "Calculate entropy for selected features");
    
    m.def("find_best_rules", &find_best_rules_py,
          "Find optimal binary splits for multiple features",
          py::arg("raw_data"), py::arg("feature_mask"), py::arg("y_index"),
          py::arg("y_class_low"), py::arg("y_class_high"),
          py::arg("sharpness_list"), py::arg("feature_names"));
}