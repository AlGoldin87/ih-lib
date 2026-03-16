#include "../include/rule_finder.hpp"
#include "../include/entropy_core.hpp"
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <limits>

// ==================== ВСПОМОГАТЕЛЬНЫЕ СТРУКТУРЫ ====================

struct VectorHash {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t seed = vec.size();
        for (size_t i = 0; i < vec.size(); i++) {
            seed ^= std::hash<int>{}(vec[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct VectorEqual {
    bool operator()(const std::vector<int>& a, const std::vector<int>& b) const {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
};

// ==================== ВНУТРЕННЯЯ ФУНКЦИЯ ЭНТРОПИИ ====================
double calc_entropy_internal(const std::vector<std::vector<int>>& data, const std::vector<int>& mask) {
    size_t n_rows = data.size();
    if (n_rows == 0) return 0.0;
    
    size_t n_cols = data[0].size();
    if (mask.size() != n_cols) return 0.0;
    
    std::unordered_map<std::vector<int>, int, VectorHash, VectorEqual> counts;
    
    for (size_t i = 0; i < n_rows; i++) {
        std::vector<int> key;
        for (size_t j = 0; j < n_cols; j++) {
            if (mask[j]) {
                key.push_back(data[i][j]);
            }
        }
        counts[key]++;
    }
    
    double h_sum = 0.0;
    double n = static_cast<double>(n_rows);
    
    for (auto it = counts.begin(); it != counts.end(); ++it) {
        double c = static_cast<double>(it->second);
        h_sum += c * std::log2(c);
    }
    
    return (n * std::log2(n) - h_sum) / n;
}

// ==================== ДИСКРЕТИЗАЦИЯ КОЛИЧЕСТВЕННЫХ ====================

struct DiscretizationInfo {
    std::vector<std::vector<int>> discrete_data;
    std::vector<double> min_vals;
    std::vector<double> max_vals;
    std::vector<double> step;
    std::vector<int> n_intervals;
};

DiscretizationInfo discretize_quantitative(
    const std::vector<std::vector<float>>& raw_data,
    const std::vector<int>& quant_indices,
    const std::vector<double>& sharpness_list) {
    
    size_t rows = raw_data.size();
    size_t n_quant = quant_indices.size();
    
    DiscretizationInfo info;
    info.discrete_data.resize(rows, std::vector<int>(n_quant));
    info.min_vals.resize(n_quant);
    info.max_vals.resize(n_quant);
    info.step.resize(n_quant);
    info.n_intervals.resize(n_quant);
    
    for (size_t f = 0; f < n_quant; f++) {
        int col = quant_indices[f];
        double sharpness = sharpness_list[f];
        
        int n_int = static_cast<int>(std::round(2.0 / sharpness));
        if (n_int < 1) n_int = 1;
        info.n_intervals[f] = n_int;
        
        double min_val = raw_data[0][col];
        double max_val = raw_data[0][col];
        for (size_t i = 1; i < rows; i++) {
            min_val = std::min(min_val, (double)raw_data[i][col]);
            max_val = std::max(max_val, (double)raw_data[i][col]);
        }
        info.min_vals[f] = min_val;
        info.max_vals[f] = max_val;
        
        double step = (max_val - min_val) / n_int;
        if (step <= 0) step = 1.0;
        info.step[f] = step;
        
        for (size_t i = 0; i < rows; i++) {
            double val = raw_data[i][col];
            int idx = static_cast<int>((val - min_val) / step);
            if (idx >= n_int) idx = n_int - 1;
            if (idx < 0) idx = 0;
            info.discrete_data[i][f] = idx;
        }
    }
    
    return info;
}

// ==================== БИНАРИЗАЦИЯ Y ====================

std::vector<int> binarize_y(
    const std::vector<std::vector<float>>& raw_data,
    int y_idx,
    double low, double high) {
    
    size_t rows = raw_data.size();
    std::vector<int> y_bin(rows);
    
    bool already_binary = true;
    for (size_t i = 0; i < rows; i++) {
        double val = raw_data[i][y_idx];
        if (val != 0.0 && val != 1.0) {
            already_binary = false;
            break;
        }
    }
    
    if (already_binary) {
        for (size_t i = 0; i < rows; i++) {
            y_bin[i] = static_cast<int>(raw_data[i][y_idx]);
        }
    } else {
        for (size_t i = 0; i < rows; i++) {
            double val = raw_data[i][y_idx];
            y_bin[i] = (val > low && val < high) ? 1 : 0;
        }
    }
    
    return y_bin;
}

// ==================== ОЦЕНКА ДЛЯ КОЛИЧЕСТВЕННОГО ПРИЗНАКА ====================

BinarySplitResult evaluate_quantitative_mask(
    const std::vector<int>& discrete_feature,
    const std::vector<int>& y_binary,
    const std::vector<int>& mask,
    int L, int R, bool inverted,
    double interval_start, double interval_end,
    const std::string& feature_name) {
    
    size_t rows = discrete_feature.size();
    std::vector<std::vector<int>> data_2d(rows, std::vector<int>(2));
    
    int class0 = 0, class1 = 0;
    
    for (size_t i = 0; i < rows; i++) {
        int x_val = discrete_feature[i];
        data_2d[i][0] = mask[x_val];
        data_2d[i][1] = y_binary[i];
        
        if (y_binary[i] == 0) class0++;
        else class1++;
    }
    
    std::vector<int> mask_x = {1, 0};
    std::vector<int> mask_y = {0, 1};
    std::vector<int> mask_xy = {1, 1};
    
    double Hx = calc_entropy_internal(data_2d, mask_x);
    double Hy = calc_entropy_internal(data_2d, mask_y);
    double Hxy = calc_entropy_internal(data_2d, mask_xy);
    
    double Ixy = Hx + Hy - Hxy;
    if (Ixy < 0 && Ixy > -1e-10) Ixy = 0.0;
    double Rxy = (Hx > 1e-10) ? Ixy / Hx : 0.0;
    
    std::ostringstream rule;
    if (inverted) {
        rule << feature_name << " NOT in [" << interval_start << ", " << interval_end << "] → class 1";
    } else {
        rule << feature_name << " in [" << interval_start << ", " << interval_end << "] → class 1";
    }
    rule << " (Rxy=" << Rxy << ")";
    
    BinarySplitResult res;
    res.Rxy = Rxy;
    res.L = L;
    res.R = R;
    res.inverted = inverted;
    res.Hx = Hx;
    res.Hy = Hy;
    res.Ixy = Ixy;
    res.class0_count = class0;
    res.class1_count = class1;
    res.interval_start = interval_start;
    res.interval_end = interval_end;
    res.feature_name = feature_name;
    res.rule_text = rule.str();
    
    return res;
}

// ==================== ПОИСК ДЛЯ КОЛИЧЕСТВЕННОГО ====================

BinarySplitResult find_best_quantitative(
    const std::vector<int>& discrete_feature,
    const std::vector<int>& y_binary,
    double min_val, double step, int n_intervals,
    const std::string& feature_name) {
    
    BinarySplitResult best;
    best.Rxy = -1.0;
    
    for (int L = 0; L < n_intervals; L++) {
        for (int R = L; R < n_intervals; R++) {
            std::vector<int> mask1(n_intervals, 0);
            for (int k = L; k <= R; k++) mask1[k] = 1;
            
            double start = min_val + L * step;
            double end = min_val + (R + 1) * step;
            
            BinarySplitResult res1 = evaluate_quantitative_mask(
                discrete_feature, y_binary, mask1, L, R, false,
                start, end, feature_name);
            
            if (res1.Rxy > best.Rxy) best = res1;
            
            std::vector<int> mask2(n_intervals, 1);
            for (int k = L; k <= R; k++) mask2[k] = 0;
            
            BinarySplitResult res2 = evaluate_quantitative_mask(
                discrete_feature, y_binary, mask2, L, R, true,
                start, end, feature_name);
            
            if (res2.Rxy > best.Rxy) best = res2;
        }
    }
    
    return best;
}

// ==================== ОЦЕНКА ДЛЯ КАТЕГОРИАЛЬНОГО ====================

BinarySplitResult evaluate_categorical_mask(
    const std::vector<int>& categorical_feature,
    const std::vector<int>& y_binary,
    const std::vector<int>& category_mask,
    const std::string& feature_name) {
    
    size_t rows = categorical_feature.size();
    std::vector<std::vector<int>> data_2d(rows, std::vector<int>(2));
    
    int class0 = 0, class1 = 0;
    
    for (size_t i = 0; i < rows; i++) {
        int cat = categorical_feature[i];
        data_2d[i][0] = category_mask[cat];
        data_2d[i][1] = y_binary[i];
        
        if (y_binary[i] == 0) class0++;
        else class1++;
    }
    
    std::vector<int> mask_x = {1, 0};
    std::vector<int> mask_y = {0, 1};
    std::vector<int> mask_xy = {1, 1};
    
    double Hx = calc_entropy_internal(data_2d, mask_x);
    double Hy = calc_entropy_internal(data_2d, mask_y);
    double Hxy = calc_entropy_internal(data_2d, mask_xy);
    
    double Ixy = Hx + Hy - Hxy;
    if (Ixy < 0 && Ixy > -1e-10) Ixy = 0.0;
    double Rxy = (Hx > 1e-10) ? Ixy / Hx : 0.0;
    
    std::ostringstream cat_list;
    cat_list << "{";
    bool first = true;
    for (size_t k = 0; k < category_mask.size(); k++) {
        if (category_mask[k]) {
            if (!first) cat_list << ",";
            cat_list << k;
            first = false;
        }
    }
    cat_list << "}";
    
    std::ostringstream rule;
    rule << feature_name << " in " << cat_list.str() << " → class 1 (Rxy=" << Rxy << ")";
    
    BinarySplitResult res;
    res.Rxy = Rxy;
    res.Hx = Hx;
    res.Hy = Hy;
    res.Ixy = Ixy;
    res.class0_count = class0;
    res.class1_count = class1;
    res.feature_name = feature_name;
    res.rule_text = rule.str();
    
    return res;
}

// ==================== ПОИСК ДЛЯ КАТЕГОРИАЛЬНОГО ====================

BinarySplitResult find_best_categorical(
    const std::vector<int>& categorical_feature,
    const std::vector<int>& y_binary,
    const std::string& feature_name) {
    
    int n_categories = 0;
    for (size_t i = 0; i < categorical_feature.size(); i++) {
        n_categories = std::max(n_categories, categorical_feature[i] + 1);
    }
    
    BinarySplitResult best;
    best.Rxy = -1.0;
    
    for (int subset = 1; subset < (1 << n_categories) - 1; subset++) {
        std::vector<int> mask(n_categories, 0);
        for (int k = 0; k < n_categories; k++) {
            if (subset >> k & 1) mask[k] = 1;
        }
        
        BinarySplitResult res = evaluate_categorical_mask(
            categorical_feature, y_binary, mask, feature_name);
        
        if (res.Rxy > best.Rxy) best = res;
    }
    
    return best;
}

// ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

std::vector<BinarySplitResult> find_best_rules(
    const std::vector<std::vector<float>>& raw_data,
    const std::vector<int>& feature_mask,
    int y_index,
    double y_class_low,
    double y_class_high,
    const std::vector<double>& sharpness_list,
    const std::vector<std::string>& feature_names) {
    
    std::vector<BinarySplitResult> results;
    
    if (raw_data.empty() || feature_mask.empty()) {
        return results;
    }
    
    // Бинаризуем Y
    std::vector<int> y_binary = binarize_y(raw_data, y_index, y_class_low, y_class_high);
    
    // Разделяем на количественные и категориальные
    std::vector<int> quant_indices;
    std::vector<double> quant_sharpness;
    std::vector<int> cat_indices;
    std::vector<std::string> quant_names;
    std::vector<std::string> cat_names;
    
    for (size_t i = 0; i < feature_mask.size(); i++) {
        if (feature_mask[i] == 1) {
            quant_indices.push_back((int)i);
            quant_sharpness.push_back(sharpness_list[quant_indices.size() - 1]);
            quant_names.push_back(feature_names[i]);
        } else if (feature_mask[i] == 2) {
            cat_indices.push_back((int)i);
            cat_names.push_back(feature_names[i]);
        }
    }
    
    // Обрабатываем количественные
    if (!quant_indices.empty()) {
        DiscretizationInfo disc_info = discretize_quantitative(raw_data, quant_indices, quant_sharpness);
        
        for (size_t f = 0; f < quant_indices.size(); f++) {
            std::vector<int> discrete_feature;
            for (size_t i = 0; i < disc_info.discrete_data.size(); i++) {
                discrete_feature.push_back(disc_info.discrete_data[i][f]);
            }
            
            BinarySplitResult res = find_best_quantitative(
                discrete_feature, y_binary,
                disc_info.min_vals[f], disc_info.step[f], disc_info.n_intervals[f],
                quant_names[f]);
            
            results.push_back(res);
        }
    }
    
    // Обрабатываем категориальные
    for (size_t f = 0; f < cat_indices.size(); f++) {
        int col = cat_indices[f];
        std::vector<int> categorical_feature;
        for (size_t i = 0; i < raw_data.size(); i++) {
            categorical_feature.push_back((int)raw_data[i][col]);
        }
        
        BinarySplitResult res = find_best_categorical(
            categorical_feature, y_binary, cat_names[f]);
        
        results.push_back(res);
    }
    
    return results;
}