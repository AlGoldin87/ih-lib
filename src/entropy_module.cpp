#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/rule_finder.hpp"
#include "../include/entropy_core.hpp"
#include <vector>
#include <string>

namespace py = pybind11;

// ==================== CALCULATE ENTROPY ====================
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

// ==================== НОВАЯ ФУНКЦИЯ FIND_BEST_RULES (новая сигнатура) ====================
py::list find_best_rules_py(
    py::array_t<int> prepared_data,
    std::vector<int> feature_mask,
    int y_index,
    std::vector<std::string> feature_names) {
    
    auto buf = prepared_data.request();
    if (buf.ndim != 2) throw std::invalid_argument("prepared_data must be 2D");
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    int* ptr = static_cast<int*>(buf.ptr);
    
    // Конвертируем numpy array в vector<vector<int>>
    std::vector<std::vector<int>> data_vec(rows, std::vector<int>(cols));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data_vec[i][j] = ptr[i * cols + j];
        }
    }
    
    // Вызываем C++ функцию с новой сигнатурой
    auto results = find_best_rules(data_vec, feature_mask, y_index, feature_names);
    
    // Конвертируем результаты в Python list of dicts
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
        d["category_mask"] = res.category_mask;  // для категориальных
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
          py::arg("prepared_data"), py::arg("feature_mask"), 
          py::arg("y_index"), py::arg("feature_names"),
          "Find optimal binary rules for classification (prepared_data must be int32 from ih-prep)");
}
